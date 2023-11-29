import re

def pre_question(question,max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ') 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


from vqaTools.vqaEval import VQAEval
from refTools.evaluation.refEvaluation import RefEvaluation

import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from tqdm import tqdm


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def vqa_eval(vqa, result_file, test_ques_path):
    vqaRes = vqa.loadRes(result_file, test_ques_path)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    # evaluate results
    vqaEval.evaluate()   

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
    print("\n")    
    
    return vqaEval


    
def collect_result(result, result_dir, filename, is_json=True, is_list=True):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
    dist.barrier()
    
    result = None
    if utils.is_main_process():   
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res) 
      
    return result    

    
def save_result(result, result_dir, filename, is_json=True, is_list=True, remove_duplicate=""):
    if is_json:
        result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.json'%filename)
        json.dump(result,open(result_file,'w'))
    else:
        result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,utils.get_rank()))
        final_result_file = os.path.join(result_dir, '%s.pth'%filename)
        torch.save(result,result_file)     
        
    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        if is_list:
            result = []
        else:
            result = {}
        for rank in range(utils.get_world_size()):
            if is_json:
                result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
                res = json.load(open(result_file,'r'))
            else:
                result_file = os.path.join(result_dir, '%s_rank%d.pth'%(filename,rank))
                res = torch.load(result_file)            
            if is_list:
                result += res
            else:
                result.update(res)
        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new  
        if is_json:                  
            json.dump(result,open(final_result_file,'w'))   
        else:            
            torch.save(result,final_result_file)     
        
        print('result file saved to %s'%final_result_file)
    dist.barrier()        
    return final_result_file



def grounding_eval(results,dets,cocos,refer,alpha,mask_size=24):
    
    correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
    correct_A, correct_B, correct_val = 0, 0, 0 
    num_A,num_B,num_val = 0,0,0
    
    for res in tqdm(results):

        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        mask = res['pred'].cuda().view(1,1,mask_size,mask_size)    
        mask = F.interpolate(mask,size = (image['height'],image['width']), mode='bicubic').squeeze()
        
        # rank detection boxes
        max_score = 0
        for det in dets[str(ref['image_id'])]:
            score = mask[int(det[1]):int(det[1]+det[3]),int(det[0]):int(det[0]+det[2])]
            area = det[2]*det[3]
            score = score.sum() / area**alpha
            if score>max_score:
                pred_box = det[:4]
                max_score = score    

        IoU_det = computeIoU(ref_box, pred_box)
        
        if ref['split']=='testA':
            num_A += 1    
            if IoU_det >= 0.5:   
                correct_A_d += 1            
        elif ref['split']=='testB':
            num_B += 1    
            if IoU_det >= 0.5:   
                correct_B_d += 1    
        elif ref['split']=='val':
            num_val += 1    
            if IoU_det >= 0.5:   
                correct_val_d += 1    
                
    eval_result = {'val_d':correct_val_d/num_val,'testA_d':correct_A_d/num_A,'testB_d':correct_B_d/num_B}        
    
    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')
        
    return eval_result    



# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union

def torch_sample(tensor, n, generator, keep_order=False):
    # 如果n==tensor.size[0] 这里变成对dim=0进行shuffle
    dim = 0
    indices = torch.randperm(tensor.size()[dim], generator=generator)[:n]
    if keep_order:
        indices = torch.sort(indices)[0]
    tensor = tensor[indices]
    return tensor

from torch.utils import data
from typing import TypeVar, Optional, Iterator
import math,random
from einops import rearrange
T_co = TypeVar('T_co', covariant=True)
class LengthBalancedDistributedSampler(data.DistributedSampler):
    def __init__(self, dataset: data.Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, num_bucket=20) -> None:
        super().__init__(dataset,num_replicas,rank,shuffle,seed,drop_last)
        self.num_bucket = num_bucket
        self.bucket_seed = 41
        assert shuffle==True
        assert drop_last==True

        with torch.no_grad():
            g = torch.Generator()
            g.manual_seed(self.seed+810975) # 避免和下面的相同
            sort_indices = torch.argsort(torch.Tensor([dataset.get_item_length(i) for i in range(len(dataset))]))
            num_each_bucket = len(sort_indices)//num_bucket
            num_samples = num_each_bucket//self.num_replicas
            self.total_size = num_samples*self.num_replicas*num_bucket
            # 去掉无法被整除的部分 使得最后的数据可以构成buckets*replicas*-1的矩阵
            sort_indices = torch_sample(sort_indices,self.total_size, generator=g, keep_order=True)
            #sort_indices = np.random.choice(sort_indices,size=self.total_size)
            
        self.sort_indices = sort_indices



    def __iter__(self) -> Iterator[T_co]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # 每一个epoch应该单独shuffle 保证每个epoch每个rank分到的数据不同
        with torch.no_grad():
            # 保持bucket之间的大小关系不变 对bucket内进行shuffle
            sort_indices = rearrange(self.sort_indices,'(B L) -> L B',B=self.num_bucket)
            sort_indices = torch_sample(sort_indices, n=sort_indices.size()[0], generator=g)
            # 取自己rank的数据 
            sort_indices = rearrange(sort_indices,'(R N) B -> R (B N)',R=self.num_replicas)
            sort_indices = sort_indices[self.rank]
        # deterministically shuffle based on epoch and seed
            sort_indices = torch_sample(sort_indices, n=sort_indices.size()[0], generator=g).tolist()  # type: ignore[arg-type]

        # remove tail of data to make it evenly divisible.
        #indices = indices[:self.total_size]
        #assert len(indices) == self.total_size

        # subsample
        #indices = indices[self.rank:self.total_size:self.num_replicas]
        #assert len(indices) == self.num_samples

        return iter(sort_indices)
        
if __name__ == '__main__':
    g = torch.Generator()
    g.manual_seed(810975) # 避免和下面的相同
    sort_indices = torch.argsort([dataset.get_item_length(i) for i in range(len(dataset))])
    num_each_bucket = len(sort_indices)//num_bucket
    num_samples = num_each_bucket//self.num_replicas
    self.total_size = num_samples*self.num_replicas*num_bucket
    # 去掉无法被整除的部分 使得最后的数据可以构成buckets*replicas*-1的矩阵
    sort_indices = torch_sample(sort_indices,self.total_size, generator=g, keep_order=True)