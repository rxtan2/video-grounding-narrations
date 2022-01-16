import os
import random
import time
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import s3dg
from args import get_args
from video_loader import HT100M_DataLoader
from loss import MILNCELoss
import sys

from metrics import compute_metrics
from youcook_loader import Youcook_DataLoader
from utils import AllGather
from utils import get_cosine_schedule_with_warmup

allgather = AllGather.apply


def main():
    args = get_args()
    assert args.eval_video_root != '' or not(args.evaluate)
    assert args.video_path != ''
    assert args.caption_root != ''
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=ngpus_per_node,
            rank=args.rank,
        )
    # create model
    
    model = s3dg.S3D(
        args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, init=args.weight_init,
    )
    
    model_dict = model.state_dict()
    pretrained_dict = {}
    if args.gpu == 0:
        print('loading model')
    checkpoint = torch.load(args.pretrained_path, map_location='cuda:{}'.format(args.gpu))
    for k in model.state_dict():
        if k in checkpoint:
            pretrained_dict[k] = checkpoint[k]
        else:
            pretrained_dict[k] = model.state_dict()[k]
    model.load_state_dict(pretrained_dict)
    
    # Freeze base encoders
    for name, param in model.named_parameters():
        if 'comma' not in name:
            param.requires_grad = False

    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.load_state_dict(net_data)
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Data loading code
    train_dataset = HT100M_DataLoader(
        csv=args.train_csv,
        video_root=args.video_path,
        caption_root=args.caption_root,
        min_time=args.min_time,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=args.crop_only,
        center_crop=args.centercrop,
        random_left_right_flip=args.random_flip,
        num_candidates=args.num_candidates,
    )
    # Test data loading code
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )

    # define loss function (criterion) and optimizer
    criterion = MILNCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, len(train_loader) * args.epochs)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint', args.checkpoint_dir)
    if args.checkpoint_dir != '' and not(os.path.isdir(checkpoint_dir)) and args.rank == 0:
        os.mkdir(checkpoint_dir)
    if args.cudnn_benchmark:
        cudnn.benchmark = True
    total_batch_size = args.world_size * args.batch_size 
    log(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ), args
    )
    scaler = torch.cuda.amp.GradScaler(True)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, train_dataset, args, scaler)
        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, checkpoint_dir, epoch + 1
            )

def train(train_loader, model, criterion, optimizer, scheduler, epoch, dataset, args, scaler):
    running_loss = 0.0
    s = time.time()
    start = time.time()
    for i_batch, sample_batch in enumerate(train_loader):
        s_step = time.time()
        batch_loss = TrainOneBatch(model, optimizer, scheduler, sample_batch, criterion, args, scaler)
        d_step = time.time() - s_step
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose and args.rank == 0:
            d = time.time() - s
            log(
                "Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f"
                % (
                    epoch + 1,
                    d,
                    args.batch_size * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.n_display,
                    optimizer.param_groups[0]['lr'],
                ), args
            )
            print("Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f" % (
                    epoch + 1,
                    d,
                    args.batch_size * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.n_display,
                    optimizer.param_groups[0]['lr'],
                ))
            running_loss = 0.0
            s = time.time()
        start_load = time.time()
    end = time.time()
    total = end - start
    if args.rank == 0:
        print('time elapsed: ' + str(total))

def TrainOneBatch(model, opt, scheduler, data, loss_fun, args, scaler):
    video = data["video"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    text = data["text"].cuda(args.gpu, non_blocking=args.pin_memory)
    idx = data["idx"].cuda(args.gpu, non_blocking=args.pin_memory)
    mask = data["mask"].cuda(args.gpu, non_blocking=args.pin_memory)
    
    text = text.view(-1, text.shape[-1])
    video = video / 255.0
    
    opt.zero_grad()
    with torch.set_grad_enabled(True):
        video_embd, text_embd = model(video, text, idx, mask)        
        loss = loss_fun(video_embd, text_embd)
    loss.backward()
    opt.step()
    scheduler.step()
    return loss.item()

def evaluate(test_loader, model, epoch, args, dataset_name):
    all_txt_embd = []
    all_video_embd = []
    model.eval()
    if args.rank == 0:  
        log('Evaluating on {}'.format(dataset_name), args)
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            text = data['text'].cuda()
            video = data['video'].float().cuda()
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd, text_embd = model(video, text)
            video_embd = video_embd.view(text_embd.shape[0], args.num_windows_test, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)
            video_embd = allgather(video_embd, args)
            text_embd = allgather(text_embd, args)
            if args.rank == 0:
                text_embd = text_embd.cpu().numpy()
                video_embd = video_embd.cpu().numpy()
                all_txt_embd.append(text_embd)
                all_video_embd.append(video_embd)
    model.train()
    if args.rank == 0:
        all_txt_embd = np.concatenate(all_txt_embd, axis=0)
        all_video_embd = np.concatenate(all_video_embd, axis=0)
        metrics = compute_metrics(np.dot(all_txt_embd, all_video_embd.T))
        log('Epoch {} results: {}'.format(epoch, metrics), args)

def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=20):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt)) 
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)
            
def save_store_checkpoint(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))

def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, 'epoch*.pth.tar'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ''

def log(output, args):
    with open(os.path.join(os.path.dirname(__file__), 'log' , args.checkpoint_dir + '.txt'), "a") as f:
        f.write(output + '\n')

if __name__ == "__main__":
    main()
