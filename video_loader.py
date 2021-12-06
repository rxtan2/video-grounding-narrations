import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import sys
import pickle
import struct
import io
import math
from PIL import Image

class HT100M_DataLoader(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(
            self,
            csv,
            video_root='',
            caption_root='',
            min_time=4.0,
            fps=16,
            num_frames=16,
            size=224,
            crop_only=False,
            center_crop=True,
            benchmark=False,
            token_to_word_path='data/dict.npy',
            max_words=20,
            num_candidates=1,
            random_left_right_flip=False,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), csv))
        self.video_root = video_root
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = 1
        self.random_flip = random_left_right_flip
        
        f = open(self.caption_root, 'video_fps.txt', 'r')
        self.vid_fps = {}
        for i, vid in enumerate(f):
            vid = vid.strip()
            vid = vid.split(',')
            self.vid_fps[vid[0]] = float(vid[-1])

        self.all_vids = list(pickle.load(open(os.path.join(self.caption_root, 'training_clips.pkl'), 'rb')))

    def __len__(self):
        return len(self.all_vids)

    def _get_text(self, video_file, caption, start, word_cap):
        cap = pd.read_csv(caption)        
        narr_ind = int(cap[cap['start'] == start].index.values[0])
        
        if self.num_candidates == 1:
            words, idx, num_words = self.words_to_ids(cap['text'].values[narr_ind])
        else:
            words = th.zeros(self.num_candidates, self.max_words, dtype=th.long)
            cap_start = self._find_nearest_candidates(cap, ind)
            for i in range(self.num_candidates):
                words[i] = self.words_to_ids(cap['text'].values[max(0, min(len(cap['text']) - 1, cap_start + i))])
        start, end = cap['start'].values[narr_ind], cap['end'].values[narr_ind]

        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time 
            
        return words, int(start), int(end), idx, num_words

    def compute_offset(self, size, sizes):
        offsets = [0]
        for i in range(size):
            offsets.append(offsets[i]+struct.unpack_from('<Q', sizes, 8*i)[0])
        return offsets
        
    def _get_video(self, video_path, start, end):
        start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def parse_bcf(self, video, start, end, fps):
        f = open(video, 'rb')
        size = struct.unpack('<Q', f.read(8))[0]
        sizes = f.read(size*8)
        offsets = self.compute_offset(size, sizes)
        end_idx = min(math.ceil(end * fps), len(offsets)-1)
        start_idx = max(0, min(math.floor(start * fps), end_idx - math.ceil(self.num_frames/fps)))
        frames = []
        for i in range(start_idx, end_idx):
            f.seek(len(offsets)*8 + offsets[i])
            data_i = f.read(offsets[i+1] - offsets[i])
            img = Image.open(io.BytesIO(data_i))
            img = np.asarray(img)
            img = th.from_numpy(img)
            frames.append(img.unsqueeze(0))
        frames = th.cat(frames, dim=0)
        frames = frames.permute(3, 0, 1, 2)
        if frames.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - frames.shape[1], 128, 128), dtype=th.uint8)
            frames = th.cat((frames, zeros), axis=1)
        f.close()
        
        return frames[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words, dtype=th.long)
        
    def _words_to_token(self, words):
        idx = 0
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we, idx, len(words)
        else:
            return th.zeros(self.max_words, dtype=th.long), 1, 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _find_nearest_candidates(self, caption, ind):
        start, end = ind, ind
        diff = caption['end'][end] - caption['start'][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
           if start == 0:
               return 0
           elif end == len(caption) - 1:
               return start - (self.num_candidates - n_candidate)
           elif caption['end'][end] - caption['start'][start - 1] < caption['end'][end + 1] - caption['start'][start]:
               start -= 1
           else:
               end += 1
           n_candidate += 1
        return start

    def __getitem__(self, idx):
        clip = self.all_vids[idx]
        word = clip[-2]
        
        video_file = clip[-1].strip()     
        fps = self.vid_fps[video_file]
        text, start, end, word_idx, num_words = self._get_text(video_file, os.path.join(self.caption_root, video_file + '.csv'), clip[0], word)
        
        if self.video_format == 'bcf':
            video_path = os.path.join(self.video_root, video_file + '.bcf')
            video = self.parse_bcf(video_path, start, end, int(fps))
        else:
            video_path = os.path.join(self.video_root, video_file + '.mp4')
            video = self._get_video(video_path, start, end)
        
        mask = th.zeros((self.max_words), dtype=th.bool)
        mask[:num_words] = True
        
        return {'video': video, 'text': text, 'idx': word_idx, 'mask': mask,}