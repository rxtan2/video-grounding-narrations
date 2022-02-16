import os
import random
import time
import sys
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn.functional as F

from args import get_args
from youcook_interactions_loader import Youcook_DataLoader
import s3dg
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import scipy
import scipy.ndimage

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    '''print('inter: ' + str(inter))
    print('union: ' + str(union))
    print('')'''
    return inter / union  # [A,B]

def main():
    args = get_args()
    print("=> loading checkpoint '{}'".format(args.checkpoint_eval))
    checkpoint = torch.load(args.checkpoint_eval)
    
    model = s3dg.S3D(
            args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path)
        
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint["state_dict"])
        
    model.eval()
    model.cuda()
   
    # Data loading code
    dataset = Youcook_DataLoader(
        args,
        data=os.path.join(os.path.dirname(__file__), 'csv/validation_youcook.csv'),
        num_clip=args.num_windows_test,
        video_root=args.eval_video_root,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=False,
        center_crop=True,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    num_valid = 0.
    num_correct = 0.
    skip = 0.
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            if i_batch % 10 == 0:
                print(i_batch)
            name = data['name'][0]
            seg = data['segment']
            text = data['text'].cuda()
            video = data['video'].float().cuda().squeeze(0)
            frame_indices = data['frame_indices']
            video = video / 255.0
            mask = data['mask'].cuda()
            word_idx = data['idx'].cuda()
            gt = data['gt']
            width = data['width']
            height = data['height']

            weights = model(video, text, mask, mode='eval')
            
            weights = torch.reshape(weights, (-1, 7, 7))
            weights = weights.unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(weights, size=(video.size(0) * video.size(2), height, width), mode='trilinear')
            upsampled = upsampled.squeeze()
            
            selected = []
            for j in range(len(upsampled)):
                if j % 1 == 0:
                    tmp = upsampled[j]
                    selected.append(tmp.unsqueeze(0))
            selected = torch.cat(selected, dim=0)
            selected = selected.cpu().numpy()
            
            # (xbr, ybr, xtl, ytl, outside, occluded)
            for j, frame_num in enumerate(list(gt.keys())):
                curr_gt = [gt[frame_num]]
                if j < len(selected):
                    curr_frame = selected[j]                
                    
                    index = np.unravel_index(curr_frame.argmax(), curr_frame.shape)
                
                valid = False
                for k in curr_gt:
                    xtl = k[0]
                    ytl = k[1]
                    xbr = k[2]
                    ybr = k[3]
                    outside = k[4]
                    occluded = k[5]
                    if outside == 1 or occluded == 1:
                        continue
                        
                    num_valid += 1.
                    
                    if index[1] >= xtl and index[1] <= xbr and index[0] >= ytl and index[0] <= ybr:
                        valid = True
                        break                   
                        
                if valid:
                    num_correct += 1.

    acc = num_correct / num_valid
    acc *= 100.
    print('localization accuracy: ' + str(acc))

if __name__ == "__main__":
    main()
