'''
 * Copyright (c)  2023, mPLUG.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from typing import Iterable

from models.distributed_gpt3 import DistributedGPT3_Retrieval
from models.modeling_distributed_gpt3 import DistributedGPT3Tokenizer
from models.vision_transformer import resize_pos_embed, resize_temporal_embed

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from dataset import create_dataset, create_sampler, create_loader
from optim import create_optimizer, create_two_optimizer
# from optim.optim_factory import check_keywords_in_name
from optim.optim_factory import get_parameter_groups
import random

import warnings
warnings.filterwarnings("ignore")


def get_parameter_groups_(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, visual_backbone_scale=False):
    parameter_group_names = {}
    parameter_group_vars = {}

    no_weight_decay = ["bias", "LayerNorm.weight"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or check_keywords_in_name(name, no_weight_decay):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if visual_backbone_scale and 'visual_encoder.' in name:
            group_name = "visual_encoder_%s" % (group_name)

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            elif visual_backbone_scale and 'visual_encoder.' in name:
                scale = 0.01
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm


def train_one_epoch(model: torch.nn.Module, tokenizer: DistributedGPT3Tokenizer,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    num_training_steps_per_epoch: int, max_norm: float = 0, update_freq: int = 1, 
                    log_writer=None, lr_scheduler=None, start_steps=None, 
                    lr_schedule_values=None, wd_schedule_values=None, beta2_values=None, args=None, 
                    global_rank=1, fp16=True):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (video, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # if data_iter_step > 10: break

        start_time = time.time()

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * \
                        param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
                if beta2_values is not None:
                    param_group["betas"][1] = beta2_values[it] if it < len(beta2_values) else beta2_values[-1]
        
        video = video.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
        idx = idx.to(device, non_blocking=True)

        if loss_scaler is None:
            if fp16:
                video = video.half()
            else:
                video = video.bfloat16()

        if loss_scaler is None:
            loss_ita = model(video, text_input, idx)
        else:
            with torch.cuda.amp.autocast():
                loss_ita = model(video, text_input, idx)

        loss = loss_ita
        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)

        all_loss_mean_value = loss_list.mean().item()
        metric_logger.update(all_loss_mean=all_loss_mean_value)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))
            if args.output_dir and args.auto_resume_iter:
                utils.auto_load_model_iter(args=args, model=model)
                continue
            else:
                exit()
    
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        end_time = time.time()

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        momentum = 1.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
            momentum = min(momentum, group["betas"][1])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(momentum=momentum)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss_ita=loss_ita.item(), head="loss")
            log_writer.update(all_rank_loss_mean=all_loss_mean_value, head="loss")
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(momentum=momentum, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(time=end_time - start_time, head="time")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
@torch.cuda.amp.autocast()
def evaluation(model, data_loader, tokenizer, device, config):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    # switch to evaluation mode
    model.eval()

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 32
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=max(64, config['max_length']), return_tensors="pt").to(
            device)
        text_embed = model.extract_text_feature(text_input)
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds, dim=0)

    sims_matrix_v2t = []
    sims_matrix_t2v = []
    for video, vid_id in data_loader:
        B, C, T, W, H = video.size()
        video = video.to(device)
        video_embed = model.extract_vision_feature(video)

        sim_matrix_v2t = torch.matmul(video_embed, text_embeds.t())
        sims_matrix_v2t.append(sim_matrix_v2t)

        sim_matrix_t2v = torch.matmul(text_embeds, video_embed.t())
        sims_matrix_t2v.append(sim_matrix_t2v)
    
    sims_matrix_v2t = torch.cat(sims_matrix_v2t, dim=0)
    sims_matrix_t2v = torch.cat(sims_matrix_t2v, dim=1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    
    torch.cuda.empty_cache()

    return (
        sims_matrix_v2t.cpu().numpy(), sims_matrix_t2v.cpu().numpy(),
    )


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Videos->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Videos
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'vid_r1': ir1,
                   'vid_r5': ir5,
                   'vid_r10': ir10,
                   'vid_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    datasets = create_dataset('video_retrieval', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    data_loader, val_loader, test_loader = create_loader(
        datasets, samplers, batch_size=[args.batch_size]* 3, 
        num_workers=[args.num_workers]*3, is_trains=[True, False, False],
        collate_fns=[None, None, None]
    )
    num_training_steps_per_epoch = len(data_loader)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # #### Model 
    
    print("Creating model")
    tokenizer = DistributedGPT3Tokenizer(config['text_decoder'])
    model = DistributedGPT3_Retrieval(config=config, tokenizer=tokenizer)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (B):', n_parameters / 1e9)

    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']
        num_patches = int(config["image_res"] * config["image_res"]/(config["visual_config"]['patch_size'] * config["visual_config"]['patch_size']))
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config["visual_config"]['embed_dim']).float())
        pos_embed = resize_pos_embed(state_dict['visual_encoder.pos_embed'], pos_embed)
        state_dict['visual_encoder.pos_embed'] = pos_embed

        num_frames = config['num_frames']
        temporal_embed = nn.Parameter(torch.zeros(1, num_frames, config["visual_config"]['embed_dim']).float())
        temporal_embed = resize_temporal_embed(state_dict['visual_encoder.temporal_embed'], temporal_embed)
        state_dict['visual_encoder.temporal_embed'] = temporal_embed

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.resume)
        print(msg)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, config["optimizer"]["weight_decay"], model.no_weight_decay(),
            visual_backbone_scale=config.get('clip_model', False)
        )
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
            model_without_ddp = model.module
            model._set_static_graph()

        optimizer = create_optimizer(
            args, model_without_ddp,
            visual_backbone_scale=config.get('clip_model', False)
        )
        loss_scaler = NativeScaler()

    print("optimizer = %s" % str(optimizer))

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        sched_type=args.lr_sched_type,
    )
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # sim_val_v2t, sim_val_t2v = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
    # val_stats = {"sim_{}".format(k): v for k, v in itm_eval(sim_val_v2t, sim_val_t2v, val_loader.dataset.txt2vid, val_loader.dataset.vid2txt).items()}
    # print(val_stats)

    # sim_test_v2t, sim_test_t2v = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
    # test_stats = {"sim_{}".format(k): v for k, v in itm_eval(sim_test_v2t, sim_test_t2v, test_loader.dataset.txt2vid, test_loader.dataset.vid2txt).items()}
    # print("Test Performance:", test_stats)

    max_epochs = args.epochs
    start_epoch = 0
    print(f"Start training for {max_epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, max_epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, 
            tokenizer,
            data_loader,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            max_norm=args.clip_grad, 
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, 
            wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq, 
            num_training_steps_per_epoch=num_training_steps_per_epoch, 
            global_rank=global_rank,
            fp16=not args.bf16,
            args=args
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == max_epochs:
                utils.save_model(
                    args=args, model=model, 
                    model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        
        # test_stats = evaluate_pt(data_loader_val, model, teacher, device, beit_like=not args.mae)
        # print(f"Val loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.1f}%")
        sim_val_v2t, sim_val_t2v = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
        val_stats = {"sim_{}".format(k): v for k, v in itm_eval(sim_val_v2t, sim_val_t2v, val_loader.dataset.txt2vid, val_loader.dataset.vid2txt).items()}
        print("Validation Performance:", val_stats)

        sim_test_v2t, sim_test_t2v = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
        test_stats = {"sim_{}".format(k): v for k, v in itm_eval(sim_test_v2t, sim_test_t2v, test_loader.dataset.txt2vid, test_loader.dataset.vid2txt).items()}
        print("Test Performance:", test_stats)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    # Resume
    parser.add_argument('--resume', default=None)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--auto_resume_iter', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.set_defaults(auto_resume_iter=True)
    
    # Other
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--enable_deepspeed',
                        action='store_true', default=False)
    parser.add_argument('--zero_stage', default=1, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please install DeepSpeed")
            exit(0)
    else:
        ds_init = None

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args_opt = utils.AttrDict(config['optimizer'])
    args_sche = utils.AttrDict(config['schedular'])

    for name, val in args_opt.items():
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, val)

    for name, val in args_sche.items():
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, val)
    
    setattr(args, "max_length", config["max_length"])
    setattr(args, "batch_size", config["batch_size"])
    setattr(args, "num_workers", config["num_workers"])
    config["image_res"] = json.load(open(config["visual_cfg"], 'r'))["img_size"]
    config["num_frames"] = config.get('num_frames', json.load(open(config["visual_cfg"], 'r'))["num_frames"])
    config['clip_model'] = json.load(open(config["visual_cfg"], 'r')).get("clip_model", False)
    config['visual_config'] = json.load(open(config["visual_cfg"], 'r'))

    if getattr(args, "log_dir") is None:
        setattr(args, "log_dir", os.path.join(args.output_dir, "tensorboard_logs"))

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config, ds_init)