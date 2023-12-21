import json
import numpy as np
import time
import logging
import os
import random
import pandas as pd
import torch

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
# from dataset.video_utils.oss_info import OSS_INFO

def decode_int32(ann):
    ann = str(ann)
    server = str(int(ann[-1]) + 1)
    id_ = "0"*(9-len(ann[:-1]))+ann[:-1]
    assert len(id_) == 9
    ann = server+"/"+id_
    return ann

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]

        if 'COCO' in ann['image']:
            dir = ann['image'].split('/')[0]+'_img'
            ann['image'] = dir + '/' + ann['image'].split('/')[-1]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            if 'COCO' in ann['image']:
                dir = ann['image'].split('/')[0]+'_img'
                ann['image'] = dir + '/' + ann['image'].split('/')[-1]
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
class nocaps_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.add_object = add_object
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        """ 
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
        """
        image_id = ann['img_id'] 
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    # file_str = bucket.get_object(file_path)
                    # file_buf = io.BytesIO()
                    # file_buf.write(file_str.read())
                    # file_buf.seek(0)
                    # file_buf = BytesIO(bucket.get_object(file_path).read())
                    # img_info = np.load(file_buf)
                    # file_buf.close()
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, image_id

class coco_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        for each in self.ann:
            filename = each["filename"]
            sentences = each["sentences"]
            filepath = each["filepath"]
            if filepath == "val2014":
                file_root = "val2014_img"
            elif filepath == "train2014":
                file_root = "train2014_img"
            else:
                file_root = filepath
            image_path = os.path.join(file_root, filename)
            gold_caption = []
            for sent in sentences:
                caption = sent["raw"]
                gold_caption.append(caption.lower())
            if self.add_object:
                object_list = each["object_label"].split("&&")
                new_object_list = list(set(object_list))
                new_object_list.sort(key=object_list.index)
                object_label = " ".join(new_object_list) 
            else:
                object_label = ""
            if is_train:
                for sent in sentences:
                    caption = sent["raw"].lower()
                    self.ann_new.append({"image": image_path, "caption": caption, "gold_caption": gold_caption, "object_label": object_label})
            else:
                self.ann_new.append({"image": image_path, "caption": sentences[0]["raw"].lower(), "gold_caption": gold_caption, "object_label": object_label})
        self.ann = self.ann_new
        del self.ann_new
            
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1] 
        object_label = ann['object_label']
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, caption, object_label, image_id, ann["gold_caption"]


class pretrain_dataset_4m(Dataset):
    def __init__(self, ann_file, transform, max_words=30, read_local_data=True, image_root="", epoch=None):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            import oss2
            bucket_name = "xxxxxx"
            auth = oss2.Auth("xxxx", "xx")
            self.bucket = oss2.Bucket(auth, "hxxxxxxxx", bucket_name)
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    time.sleep(0.1)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, caption



class pretrain_dataset_laion(Dataset):
    def __init__(self, ann_file, transform, max_words=800, read_local_data=True, image_root="", epoch=None):
        self.ann = []
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            bucket_name = "nlp-mind"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)
        self.ann = open("xxxxxxxxxxxx", "r").readlines()

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        # ann = self.ann[index]
        ann = self.ann[index].strip()
        # if type(ann['caption']) == list:
        #     caption = pre_caption(random.choice(ann['caption']), self.max_words)
        # else:
        #     caption = pre_caption(ann['caption'], self.max_words)

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                try:
                    # image = self.bucket.get_object("mm_feature/" + ann['image'])
                    # image = BytesIO(image.read())
                    # image = Image.open(image).convert('RGB')
                    # image = self.transform(image)
                    image = self.bucket.get_object(ann)
                    #print("image:", image)
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                    #print("image:", image)
                    try:
                        ann = ann.replace("jpg", "json").replace("jpeg", "json")
                        caption = self.bucket.get_object(ann)
                        caption = caption.read().decode("utf-8")
                        caption = json.loads(caption)['caption']
                    except json.JSONDecodeError:
                        print("Get caption:{} from oss failed, retry.".format(ann))
                        index = 0 if index == (len(self) - 1) else index + 1
                        ann = self.ann[index].strip()
                        time.sleep(0.1)
                        #print("index:", index)
                        continue
                    caption = pre_caption(caption, self.max_words)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann))
                    print("Get ann:{} from oss failed, retry.".format(ann))
                    time.sleep(0.1)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index].strip()
                    continue
                break

        return image, caption



class pretrain_dataset_redbook(Dataset):
    
    def __init__(self, ann_file, transform, max_words=800, read_local_data=True, image_root="", epoch=None):
        
        self.ann = []
        for pt_file in ann_file:
            data = torch.load(pt_file)
            self.ann.append(data)
        self.ann = pd.concat(self.ann)

        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            bucket_name = "mm-chatgpt"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)

        # self.prompt = "<doc>\n这个图片的关键词包含：\n{}\n请给这张图片写一个富含知识性的文案：{}\n</doc>"
        # title 24
        # desc 256
        # tag 32

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann.iloc[index]
        image_path = os.path.join('redbook/image_data', random.choice(ann['images']).split('/')[-1])
        # image_path = os.path.join('redbook/image_data', ann['cover_image'])

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                image_path = os.path.join('redbook/image_data', random.choice(ann['images']).split('/')[-1])
                try:
                    image = self.bucket.get_object(image_path)
                    #print("image:", image)
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except Exception as e:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann))
                    # print("Get ann:{} from oss failed, retry.".format(ann))
                    print(e)
                    time.sleep(0.1)
                    index = random.randint(0, len(self) - 1)
                    ann = self.ann.iloc[index]
                    continue
                break

        tag = "；".join(ann['tag']) if len(ann['tag']) > 0 else None
        title = ann['title'] if len(ann['title']) > 0 else None
        desc = ann['desc']

        # caption = ""
        # if tag is not None:
        #     caption += "这个图片的关键词包含：\n{}\n".format(tag[:32])
        # if title is not None:
        #     caption += "请给这张图片写一个标题：\n{}\n".format(title[:24])
        # if desc is not None:
        #     caption += "请给这张图片配一个富含生活气息的文案：\n{}\n".format(desc[:256])
        if random.random() <= 0.25 and tag is not None:
            prompt = "这个图片的关键词包含：\n"
            caption = tag[:32]
        elif 0.25 < random.random() <= 0.5 and title is not None:
            prompt = "请给这张图片写一个标题：\n"
            caption = title[:36]
        else:
            prompt = "请给这张图片配一个富含生活气息的文案：\n"
            caption = desc[:224]

        return image, prompt, caption




class pretrain_dataset_baike(Dataset):
    
    def __init__(self, ann_file, transform, max_words=800, read_local_data=True, image_root="", epoch=None):
        
        self.ann = []
        if len(ann_file) > 1:
            for pt_file in ann_file:
                data = pd.read_pickle(pt_file)
                self.ann.append(data)
            self.ann = pd.concat(self.ann)
        else:
            self.ann = pd.read_pickle(ann_file[0])

        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            bucket_name = "mm-chatgpt"
            auth = oss2.Auth(OSS_INFO[bucket_name]["AK"], OSS_INFO[bucket_name]["SK"])
            self.bucket = oss2.Bucket(auth, OSS_INFO[bucket_name]["ENDPOINT"], bucket_name)

        self.prompt = "这个图片的关键词包含：\n{}\n请给这张图片写一个富含知识性的文案：\n{}\n"

    def __len__(self):
        return len(self.ann)


    def __getitem__(self, index):

        ann = self.ann.iloc[index]
        image_path = os.path.join('baidubaike/image_data', "{}.jpg".format(ann['image_url'].split('/')[-1]) )

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                image_path = os.path.join('baidubaike/image_data', "{}.jpg".format(ann['image_url'].split('/')[-1]) )
                try:
                    image = self.bucket.get_object(image_path)
                    #print("image:", image)
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                    #print("image:", image)
                except Exception as e:
                    print(image_path, e)
                    #logging.info("Get image:{} from oss failed, retry.".format(ann))
                    # print("Get ann:{} from oss failed, retry.".format(ann))
                    time.sleep(0.1)
                    index = random.randint(0, len(self) - 1)
                    ann = self.ann.iloc[index]
                    continue
                break

        # caption = self.prompt.format(ann['name'][:16], ann['text'][:200])
        if random.random() <= 0.25:
            prompt = "这个图片的关键词包含：\n"
            caption = ann['name'][:16]
        else:
            prompt = "请给这张图片写一个富含知识性的文案：\n"
            caption = ann['text'][:224]

        return image, prompt, caption
