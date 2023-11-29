import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import (
    re_train_dataset, re_eval_dataset, pretrain_dataset_4m, 
    coco_dataset, nocaps_dataset,
)
from dataset.nlvr_dataset import nlvr_dataset
from dataset.utils import LengthBalancedDistributedSampler
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import build_uni_training_dataset,build_vg_dataset

from dataset.randaugment import RandomAugment

# Video Stuff
from dataset.video_utils import video_transforms, volume_transforms
from dataset.video_pretrain_dataset import pretrain_dataset_video, pretrain_eval_dataset_video
from dataset.video_downstream_datasets import (
    video_retrieval_dataset_train, video_retrieval_dataset_eval,
    video_qa_dataset, video_caption_dataset, video_cls_dataset
)
from dataset.dataset_folder import ImageFolder, ImageNet21K

from dataset.pretrain_transforms import DataAugmentationForPretrain


def create_dataset(dataset, config, epoch=None):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])

    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    
    video_pretrain_transform = transforms.Compose([
        # video_transforms.Resize((int(config['image_res'] * 1.14), int(config['image_res'] * 1.14))),
        # video_transforms.ShortSideScale(int(config['image_res'] * 256 // 224)),
        video_transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation="bicubic"),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.TemporalConsistentRandomAugment(N = 2, M = 5, augs = ['Identity', 'Contrast', 'Brightness', 
            'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        # video_transforms.RandomCrop(config['image_res']),
        volume_transforms.ClipToTensor(channel_nb=3),
        video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    video_train_transform = transforms.Compose([
        video_transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation="bicubic"),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.TemporalConsistentRandomAugment(N = 2, M = 5, augs = ['Identity', 'Contrast', 'Brightness', 
            'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        volume_transforms.ClipToTensor(channel_nb=3),
        video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    video_test_transform = transforms.Compose([
        video_transforms.Resize((config['image_res'], config['image_res'])),
        volume_transforms.ClipToTensor(channel_nb=3),
        video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    if dataset=='pretrain':
        dataset = pretrain_dataset_4m(config['train_file'], pretrain_transform, read_local_data=config['read_local_data'])
        return dataset


    elif dataset == "pretrain_video":
        dataset = pretrain_dataset_video(config['train_file'], video_pretrain_transform, config["train_video_root"], 
            num_frames=config["num_frames"], read_local_data=config["read_local_data"])
        return dataset

    elif dataset == "video_cls":
        train_dataset = video_cls_dataset(config['train_file'], video_train_transform, config["video_root"], 
            num_frames=config["num_frames"], read_local_data=config["read_local_data"], train=True)
        val_dataset = video_cls_dataset(config['val_file'], video_test_transform, config["video_root"], 
            num_frames=config["num_frames"], read_local_data=config["read_local_data"], train=False)
        test_dataset = video_cls_dataset(config['test_file'], video_test_transform, config["video_root"], 
            num_frames=config["num_frames"], read_local_data=config["read_local_data"], train=False)
        return train_dataset, val_dataset, test_dataset

    elif dataset == "video_caption":
        train_dataset = video_caption_dataset(config["train_file"], video_train_transform, config["video_root"],
            num_frames=config["num_frames"], read_local_data=config["read_local_data"], split='train')
        val_dataset = video_caption_dataset(config["val_file"], video_test_transform, config["video_root"],
            num_frames=config["num_frames"], read_local_data=config["read_local_data"], split='test')
        test_dataset = video_caption_dataset(config["test_file"], video_test_transform, config["video_root"],
            num_frames=config["num_frames"], read_local_data=config["read_local_data"], split='test')
        return train_dataset, val_dataset, test_dataset
    

    elif dataset == "video_retrieval":
        train_dataset = video_retrieval_dataset_train(config["train_file"], video_train_transform, config["video_root"],
            num_frames=config["num_frames"], has_multi_vision_gt=config.get("has_multi_vision_gt", False), 
            is_paragraph_retrieval=config.get("is_paragraph_retrieval", False), read_local_data=config["read_local_data"])
        val_dataset = video_retrieval_dataset_eval(config["val_file"], video_test_transform, config["video_root"],
            num_frames=config["num_frames"], has_multi_vision_gt=config.get("has_multi_vision_gt", False), 
            is_paragraph_retrieval=config.get("is_paragraph_retrieval", False), read_local_data=config["read_local_data"])
        test_dataset = video_retrieval_dataset_eval(config["test_file"], video_test_transform, config["video_root"],
            num_frames=config["num_frames"], has_multi_vision_gt=config.get("has_multi_vision_gt", False), 
            is_paragraph_retrieval=config.get("is_paragraph_retrieval", False), read_local_data=config["read_local_data"])
        return train_dataset, val_dataset, test_dataset



def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n

def nocaps_collate_fn(batch):
    image_list, image_id_list = [], []
    for image, image_id in batch:
        image_list.append(image)
        image_id_list.append(image_id)
    return torch.stack(image_list,dim=0), image_id_list

def coco_collate_fn(batch):
    image_list, caption_list, object_labels, image_id_list, gold_caption_list = [], [], [], [], []
    for image, caption, object_label, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        object_labels.append(object_label)
    return torch.stack(image_list,dim=0), caption_list, object_labels, image_id_list, gold_caption_list


def create_sampler(datasets, shuffles, num_tasks, global_rank, balance_sample=False):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        if balance_sample == True and shuffle == True:
            sampler = LengthBalancedDistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle, drop_last=True)
        else:  
            sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    


def create_pretrain_sampler(datasets, shuffle, num_tasks, global_rank):
    samplers = []
    for dataset in datasets:
        sampler = torch.utils.data.DistributedSampler(dataset[0], num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_pretrain_loader(datasets, samplers, video_batch_size, num_workers, is_train, collate_fn, suffix=True):
    loaders = []
    data_types = []
    nums = list(range(len(datasets)))
    for dataset, sampler, i in zip(datasets, samplers, nums):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        data_type = dataset[1]
        loader = DataLoader(
            dataset[0],
            batch_size=video_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
        if suffix:
            data_types.append(data_type + "_" + str(i))
        else:
            data_types.append(data_type)

    return loaders, data_types
