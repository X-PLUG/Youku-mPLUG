import json
import numpy as np
import logging
import os
import random

import torch
from torch.utils.data import Dataset
import torchvision

import pandas as pd
from io import BytesIO
from dataset.utils import pre_caption, load_jsonl

from .video_utils.utils import read_frames_decord, read_from_tar
# from .video_utils.oss_info import OSS_INFO

class pretrain_dataset_video(Dataset):
    def __init__(self, ann_file, transform, video_path, num_frames=8, max_words=30, read_local_data=True):
        self.ann = []
        for f in ann_file:
            if '.csv' in f:
                df = pd.read_csv(f)
                self.ann += [{'video_id': video_id, 'caption': title} for video_id, title in zip(df['video_id:FILE'], df['title'])]
            else:
                self.ann += json.load(open(f,'r'))
        self.video_path = video_path
        self.max_words = max_words
        self.num_frames = num_frames
        self.transform = transform

        self.read_local_data = read_local_data
        if not self.read_local_data:
            import oss2
            bucket_name = "xdp-expriment"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)
        else:
            self.bucket = None

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        num_retries = 20  # skip error videos

        for _ in range(num_retries):
            ann = self.ann[index]
            video_id = ann['video_id']
            text = ann['caption']

            start_time = ann.get('start_time', None)
            end_time = ann.get('end_time', None)

            # read with retries
            for i in range(3):
                if self.read_local_data:
                    video_path = os.path.join(self.video_path, video_id)
                    try:
                        if video_id.endswith(".tar"):
                            video_buffer = read_from_tar(video_path, self.bucket)
                        else:
                            video_buffer = video_path
                        video_array = read_frames_decord(video_buffer, num_frames=self.num_frames, sample='rand',
                            start_time=start_time, end_time=end_time)
                    except Exception as e:
                        print(e)
                        video_array = None
                        continue
                else:
                    try:
                        video_path = os.path.join(self.video_path, video_id)
                        if video_id.endswith(".tar"):
                            video_buffer = read_from_tar(video_path, self.bucket)
                        else:
                            video_buffer = BytesIO(self.bucket.get_object(video_path).read())
                        video_array = read_frames_decord(video_buffer, num_frames=self.num_frames, sample='rand',
                            start_time=start_time, end_time=end_time)
                    except Exception as e:
                        print(e)
                        video_array = None
                        continue

                if video_array is not None:
                    break

            # Select a random video if the current video was not able to access.
            if video_array is None:
                index = random.randint(0, len(self) - 1)
                continue
            else:
                if self.transform:
                    video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)
                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        return video_array, caption


class pretrain_eval_dataset_video(Dataset):
    def __init__(self, ann_file, transform, video_path, num_frames=8, max_words=30, read_local_data=True):
        if '.csv' in ann_file:
            df = pd.read_csv(ann_file)
            self.ann = [{'video_id': video_id, 'caption': title} for video_id, title in zip(df['video_id:FILE'], df['title'])]
        else:
            self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.video_path = video_path
        self.max_words = max_words
        self.num_frames = num_frames

        self.read_local_data = read_local_data

        self.text = []
        self.video = []
        self.txt2vid = {}
        self.vid2txt = {}

        txt_id = 0
        for vid_id, ann in enumerate(self.ann):
            self.video.append(ann['video_id'])
            self.vid2txt[vid_id] = []
            for i, caption in enumerate([ann['caption']]):
                self.text.append(pre_caption(caption, self.max_words))
                self.vid2txt[vid_id].append(txt_id)
                self.txt2vid[txt_id] = vid_id
                txt_id += 1

        if not self.read_local_data:
            import oss2
            bucket_name = "nlp-mind"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)
        else:
            self.bucket = None

    def __len__(self):
        return len(self.video)

    def __getitem__(self, index):
        num_retries = 10  # skip error videos

        for _ in range(num_retries):
            ann = self.ann[index]
            video_id = ann['video_id']
            text = ann['caption']

            # read with retries
            for i in range(5):
                if self.read_local_data:
                    video_path = os.path.join(self.video_path, video_id)
                    try:
                        if video_id.endswith(".tar"):
                            video_buffer = read_from_tar(video_path, self.bucket)
                        else:
                            video_buffer = video_path
                        video_array = read_frames_decord(video_buffer, num_frames=self.num_frames, sample='middle')
                    except Exception as e:
                        print(e)
                        video_array = None
                        continue
                else:
                    try:
                        video_path = os.path.join(self.video_path, video_id)
                        if video_id.endswith(".tar"):
                            video_buffer = read_from_tar(video_path, self.bucket)
                        else:
                            video_buffer = BytesIO(self.bucket.get_object(video_path).read())
                        video_array = read_frames_decord(video_buffer, num_frames=self.num_frames, sample='middle')
                    except Exception as e:
                        print(e)
                        video_array = None
                        continue

                if video_array is not None:
                    break

            # Select a random video if the current video was not able to access.
            if video_array is None:
                index = random.randint(0, len(self) - 1)
                continue
            else:
                if self.transform:
                    video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)
                break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        return video_array, caption
