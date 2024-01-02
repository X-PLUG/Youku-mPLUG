import json
import numpy as np
import time
import logging
import os
from io import BytesIO

from torch.utils.data import Dataset

from dataset.utils import pre_caption, pre_question, load_jsonl
from .video_utils.utils import read_frames_decord, read_frames_gif
# from .video_utils.oss_info import OSS_INFO
import pandas as pd
import ast


'''
Video Retrieval Datasets

jsonl format [
    {"clip_name": str, "caption": Any[List[str], str]},
    ....
]
'''

def preprocess_para_retrieval_data(anno_list):
    processed_anno_list = []
    for d in anno_list:
        d["caption"] = " ".join(d.pop("caption"))
        processed_anno_list.append(d)
    return processed_anno_list


class video_retrieval_dataset_train(Dataset):
    def __init__(self, ann_file, transform, video_root, num_frames=4, max_words=80, has_multi_vision_gt=False,
                 is_paragraph_retrieval=False, read_local_data=True, has_extension=True):
        if '.csv' in ann_file:
            df = pd.read_csv(ann_file)
            self.ann = [{'clip_name': clip_name, 'caption': caption} for clip_name, caption in zip(df['clip_name:FILE'], df['caption'])]
        else:
            self.ann = load_jsonl(ann_file)
        self.transform = transform
        self.video_root = video_root
        self.max_words = max_words
        self.num_frames = num_frames
        self.has_multi_vision_gt = has_multi_vision_gt
        self.match_ids = {}

        if is_paragraph_retrieval:
            self.ann = preprocess_para_retrieval_data(self.ann)
        
        self.has_extension = has_extension

        n = 0
        for ann in self.ann:
            key = ann['caption'] if has_multi_vision_gt else ann['clip_name']
            if key not in self.match_ids.keys():
                self.match_ids[key] = n
                n += 1

        self.read_local_data = read_local_data
        if not self.read_local_data:
            import oss2
            bucket_name = "xke-repo"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        video_path = os.path.join(self.video_root, ann['clip_name']) 
        if not self.has_extension: video_path += '.mp4'

        if self.read_local_data:
            while True:
                try:
                    if not os.path.exists(video_path):
                        video_path = os.path.join(self.video_root, ann['clip_name'])
                        if not self.has_extension: video_path += '.avi'
                    video = read_frames_decord(video_path, num_frames=self.num_frames, sample='rand')
                except:
                    time.sleep(0.01)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    video_path = os.path.join(self.video_root, ann['clip_name'])
                    if not self.has_extension: video_path += '.mp4'
                    continue
                break
        else:
            while True:
                try:
                    if not self.bucket.object_exists(video_path):
                        video_path = os.path.join(self.video_root, ann['clip_name'] + '.avi')
                    video = self.bucket.get_object(video_path)
                    video = BytesIO(video.read())
                    video = read_frames_decord(video, num_frames=self.num_frames, sample='rand')
                except:
                    time.sleep(0.01)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    video_path = os.path.join(self.video_root, ann['clip_name'] + '.mp4')
                    continue
                break

        if self.transform:
            video = self.transform(video) # (T, C, H, W) -> (C, T, H, W)

        caption = pre_caption(ann['caption'], self.max_words)
        key = ann['caption'] if self.has_multi_vision_gt else ann['clip_name']

        return video, caption, self.match_ids[key]


class video_retrieval_dataset_eval(Dataset):
    def __init__(self, ann_file, transform, video_root, num_frames=8, max_words=80, has_multi_vision_gt=False,
                 is_paragraph_retrieval=False, read_local_data=True, has_extension=True):
        if '.csv' in ann_file:
            df = pd.read_csv(ann_file)
            self.ann = [{'clip_name': clip_name, 'caption': caption} for clip_name, caption in zip(df['clip_name:FILE'], df['caption'])]
        else:
            self.ann = load_jsonl(ann_file)
        self.transform = transform
        self.video_root = video_root
        self.max_words = max_words
        self.has_multi_vision_gt = has_multi_vision_gt
        self.num_frames = num_frames

        if is_paragraph_retrieval:
            self.ann = preprocess_para_retrieval_data(self.ann)

        self.has_extension = has_extension

        self.text = []
        self.video = []
        self.txt2vid = {}
        self.vid2txt = {}

        self.read_local_data = read_local_data
        if not self.read_local_data:
            import oss2
            bucket_name = "xke-repo"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)

        if self.has_multi_vision_gt:
            """each text may have multiple ground_truth image, e.g., ssv2"""
            vid_id = 0
            for txt_id, ann in enumerate(self.ann):
                self.text.append(pre_caption(ann["caption"], self.max_words))
                self.txt2vid[txt_id] = []
                _videos = ann["clip_name"] if isinstance(ann["clip_name"], list) else [ann["clip_name"], ]
                for i, video in enumerate(_videos):
                    self.video.append(video)
                    self.txt2vid[txt_id].append(vid_id)
                    self.vid2txt[vid_id] = txt_id
                    vid_id += 1
        else:
            """each image may have multiple ground_truth text， e.g., COCO and Flickr30K"""
            txt_id = 0
            for vid_id, ann in enumerate(self.ann):
                self.video.append(ann["clip_name"])
                self.vid2txt[vid_id] = []
                _captions = ann["caption"] if isinstance(ann["caption"], list) else [ann["caption"], ]
                for i, caption in enumerate(_captions):
                    self.text.append(pre_caption(caption, self.max_words))
                    self.vid2txt[vid_id].append(txt_id)
                    self.txt2vid[txt_id] = vid_id
                    txt_id += 1

        self.anno_list = [dict(clip_name=e) for e in self.video]

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):

        ann = self.anno_list[index]
        video_path = os.path.join(self.video_root, ann['clip_name'])
        if not self.has_extension: video_path += '.mp4'
        if self.read_local_data:
            if not os.path.exists(video_path):
                video_path = os.path.join(self.video_root, ann['clip_name'])
                if not self.has_extension: video_path += '.avi'
            video = read_frames_decord(video_path, num_frames=self.num_frames, sample='middle')
        else:
            while True:
                try:
                    if not self.bucket.object_exists(video_path):
                        video_path = os.path.join(self.video_root, ann['clip_name'] + '.avi')
                    video = self.bucket.get_object(video_path)
                    video = BytesIO(video.read())
                    video = read_frames_decord(video, num_frames=self.num_frames, sample='middle')
                except:
                    time.sleep(0.01)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    video_path = os.path.join(self.video_root, ann['clip_name'] + '.mp4')
                    continue
                break
        if self.transform:
            video = self.transform(video) # (T, C, H, W) -> (C, T, H, W)

        return video, index


'''
Video Question Answering Dataset (Open-ended) in vqa format

jsonl format [
    {"video_id": str, "question": str, "answer": str},
    xxxx
]
'''

class video_qa_dataset(Dataset):
    def __init__(self, ann_file, transform, video_root, num_frames=16, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list='', read_local_data=True):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += load_jsonl(f)
        self.transform = transform
        self.video_root = video_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.num_frames = num_frames
        self.read_local_data = read_local_data
        if not self.read_local_data:
            bucket_name = "xke-repo"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)

        if split == 'test':
            self.max_ques_words = 50  # do not limit question length during test
            if answer_list.split('.')[-1] == 'json':
                self.answer_list = list(json.load(open(answer_list, 'r')).keys())
            else:
                self.answer_list = list(set([x['answer'] for x in load_jsonl(answer_list)]))
        for idx, ann in enumerate(self.ann):
            ann['question_id'] = idx

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        video_path = os.path.join(self.video_root, ann['video_id'] + '.mp4')

        if self.split == 'train':
            question = pre_question(ann['question'], self.max_ques_words)
            answers = [ann['answer']]
            answers = [answer + self.eos for answer in answers]
            weights = [1]
            if self.read_local_data:
                while True:
                    try:
                        if 'tumblr' == ann['video_id'].split('_')[0]:
                            video_path = os.path.join(self.video_root, ann['video_id'] + '.gif')
                            video_array = read_frames_gif(video_path, num_frames=self.num_frames, sample='rand')
                        else:
                            if not os.path.exists(video_path):
                                video_path = os.path.join(self.video_root, ann['video_id'] + '.avi')
                            video_array = read_frames_decord(video_path, num_frames=self.num_frames, sample='rand')
                    except:
                        index = 0 if index == (len(self) - 1) else index + 1
                        ann = self.ann[index]
                        video_path = os.path.join(self.video_root, ann['video_id'] + '.mp4')
                        print("Failed to load example")
                        continue
                    break
            else:
                while True:
                    try:
                        if not self.bucket.object_exists(video_path):
                            video_path = os.path.join(self.video_root, ann['video_id'] + '.avi')
                        video_array = self.bucket.get_object(video_path)
                        video_array = BytesIO(video_array.read())
                        video_array = read_frames_decord(video_array, num_frames=self.num_frames, sample='rand')
                    except:
                        index = 0 if index == (len(self) - 1) else index + 1
                        ann = self.ann[index]
                        video_path = os.path.join(self.video_root, ann['video_id'] + '.mp4')
                        continue
                    break
            if self.transform:
                video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)
                
            return video_array, question, answers, weights
        
        elif self.split == 'test':
            question = pre_question(ann['question'], self.max_ques_words)
            question_id = ann['question_id']
            if self.read_local_data:
                if 'tumblr' == ann['video_id'].split('_')[0]:
                    video_path = os.path.join(self.video_root, ann['video_id'] + '.gif')
                    video_array = read_frames_gif(video_path, num_frames=self.num_frames, sample='middle')
                else:
                    if not os.path.exists(video_path):
                        video_path = os.path.join(self.video_root, ann['video_id'] + '.avi')
                    video_array = read_frames_decord(video_path, num_frames=self.num_frames, sample='middle')
            else:
                while True:
                    try:
                        if not self.bucket.object_exists(video_path):
                            video_path = os.path.join(self.video_root, ann['video_id'] + '.avi')
                        video_array = self.bucket.get_object(video_path)
                        video_array = BytesIO(video_array.read())
                        video_array = read_frames_decord(video_array, num_frames=self.num_frames, sample='middle')
                    except:
                        index = 0 if index == (len(self) - 1) else index + 1
                        ann = self.ann[index]
                        video_path = os.path.join(self.video_root, ann['video_id'] + '.mp4')
                        continue
                    break
            if self.transform:
                video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)
                
            return video_array, question, question_id


'''
Video Captioning Dataset

jsonl format [
    {"video_id": str, "caption": str}, or {"video_id": str, "golden_caption": List[str]}
    xxxx
]
'''
    
class video_caption_dataset(Dataset):
    def __init__(self, ann_file, transform, video_root, num_frames=16,
                split='train', max_words=80, read_local_data=True, has_extension=True):
        if '.csv' in ann_file:
            df = pd.read_csv(ann_file)
            if split == 'train':
                self.ann = [{'video_id': video_id, 'caption': golden_caption} for video_id, golden_caption in zip(df['video_id:FILE'], df['golden_caption'])]
            else:
                if df['golden_caption'].isnull().all():
                    self.ann = [{'video_id': video_id, 'golden_caption': golden_caption} for video_id, golden_caption in zip(df['video_id:FILE'], df['golden_caption'])]
                else:
                    self.ann = [{'video_id': video_id, 'golden_caption': ast.literal_eval(golden_caption)} for video_id, golden_caption in zip(df['video_id:FILE'], df['golden_caption'])]
        else:
            self.ann = load_jsonl(ann_file)
        self.transform = transform
        self.max_words = max_words
        self.video_root = video_root
        self.split = split
        self.num_frames = num_frames
        self.read_local_data = True
        self.has_extension = has_extension

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        
        ann = self.ann[index]
        
        if self.split == 'train':
            video_path = os.path.join(self.video_root, ann['video_id'])
            if not self.has_extension: video_path += '.mp4'
            while True:
                if not os.path.exists(video_path):
                    video_path = os.path.join(self.video_root, ann['video_id'])
                    if not self.has_extension: video_path += '.avi'
                try:
                    video_array = read_frames_decord(video_path, num_frames=self.num_frames, sample='rand')
                except:
                    time.sleep(0.01)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    video_path = os.path.join(self.video_root, ann['video_id'])
                    if not self.has_extension: video_path += '.mp4'
                    continue
                break
            caption = pre_caption(ann['caption'], 80)
            if self.transform:
                video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)

            return video_array, caption
            
        else:
            video_id = ann['video_id']
            video_path = os.path.join(self.video_root, ann['video_id'])
            if not self.has_extension: video_path += '.mp4'
            if not os.path.exists(video_path):
                video_path = os.path.join(self.video_root, ann['video_id'])
                if not self.has_extension: video_path += '.avi'
            video_array = read_frames_decord(video_path, num_frames=self.num_frames, sample='middle')
            golden_captions = [x.lower() for x in ann['golden_caption']]
            
            if self.transform:
                video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)

            return video_array, video_id, golden_captions



'''
Video Classification Dataset

jsonl format [
    {"video_id": str, "caption": str, "label": int}
    xxxx
]
'''
    
class video_cls_dataset(Dataset):
    def __init__(self, ann_file, transform, video_root, num_frames=16, max_words=80, train=True, read_local_data=True):
        self.label2idx = json.load(open('classname.json', 'r'))
        if '.csv' in ann_file:
            df = pd.read_csv(ann_file)
            if df['label'].isnull().all():
                self.ann = [{'video_id': video_id, 'caption': title, 'label': -1} for video_id, title in zip(df['video_id:FILE'], df['title'])]
            else:
                self.ann = [{'video_id': video_id, 'caption': title, 'label': self.label2idx[label]} for video_id, title, label in zip(df['video_id:FILE'], df['title'], df['label'])]
        else:
            self.ann = load_jsonl(ann_file)
        self.transform = transform
        self.max_words = max_words
        self.video_root = video_root
        self.num_frames = num_frames
        self.read_local_data = True
        self.train = train
        self.idx2label = {v:k for k,v in self.label2idx.items()}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        
        ann = self.ann[index]
        
        video_path = os.path.join(self.video_root, ann['video_id'])
        while True:
            try:
                video_array = read_frames_decord(video_path, num_frames=self.num_frames, sample='rand' if self.train else 'middle')
            except:
                time.sleep(0.01)
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
                video_path = os.path.join(self.video_root, ann['video_id'])
                continue
            break
        caption = pre_caption(ann['caption'], 80)
        label = int(ann['label'])
        if self.transform:
            video_array = self.transform(video_array) # (T, C, H, W) -> (C, T, H, W)

        return video_array, caption, label