'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vision_transformer import (
    TimeSformer, VisionTransformer, LayerNormWithForceFP32, 
    _convert_pretrained_vit, AttentionPool
)
from models.eva_vit import create_eva_vit_g, interpolate_pos_embed
from models.modeling_distributed_gpt3 import GPT3Config, DistributedGPT3
from models.distributed_utils import all_gather

import torch
import torch.nn.functional as F
from torch import nn

import timm
from timm.models.layers import trunc_normal_

import json
import numpy as np
import random
import math

from einops import rearrange, repeat

class DistributedGPT3_Pretrain(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
        text_cfg = GPT3Config.from_json_file(config['text_cfg'])

        self.visual_encoder = TimeSformer(
            img_size=visual_cfg['img_size'], num_frames=visual_cfg['num_frames'],
            patch_size=visual_cfg['patch_size'], 
            embed_dim=visual_cfg['embed_dim'], depth=visual_cfg['depth'], 
            num_heads=visual_cfg['num_heads'],
            mlp_ratio=visual_cfg['mlp_ratio'], qkv_bias=True,
            norm_layer=partial(LayerNormWithForceFP32, eps=1e-6),
            init_std=0.015, grad_ckpt=visual_cfg.get('grad_ckpt', True),
            drop_path_rate=visual_cfg.get('drop_path', False),
            stop_grad_conv1=visual_cfg.get('stop_grad_conv1', False),
            use_shared_rel_pos_bias=visual_cfg.get('use_shared_rel_pos_bias', False),
            use_abs_pos_emb=visual_cfg.get('use_abs_pos_emb', True),
            init_values=visual_cfg.get('layer_scale_init_value', 0),
            postnorm=visual_cfg.get('postnorm', False),
            clip_model=visual_cfg.get('clip_model', False)
        )

        pretrained_vit = visual_cfg.get("pretrained_ckpt", None) # timm/xxxxxx
        if pretrained_vit is not None:
            if pretrained_vit.startswith("timm"):
                pretrained_vit_name = pretrained_vit.split("/")[-1]
                pt_weights = timm.create_model(
                    pretrained_vit_name,
                    pretrained=True,
                ).state_dict()
                pt_weights = _convert_pretrained_vit(pt_weights)
            elif pretrained_vit.startswith("clip"):
                pretrained_vit_name = "/".join(pretrained_vit.split("/")[1:])
                pt_weights = torch.load(pretrained_vit_name, map_location='cpu')
                pt_weights = _convert_pretrained_vit(pt_weights)

            msg = self.visual_encoder.load_state_dict(pt_weights, strict=False)
            print("Initialize Vision Encoder from CKPT {}".format(pretrained_vit_name))
            print(msg)

        
        megatron_cfg = config['megatron_cfg']
        # megatron_cfg['world_size'] = torch.distributed.get_world_size()
        # megatron_cfg['world_size'] = 1
        self.text_decoder = DistributedGPT3(
            model_dir=config['text_decoder'],
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            path_load_tag='model', 
            megatron_cfg=megatron_cfg, 
            checkpoint_model_parallel_size=1 if text_cfg.num_hidden_layers < 40 else 8
        )

        if config.get('freeze_vit', False):
            for name, param in self.visual_encoder.named_parameters():
                if not any([x in name for x in ['time', 'temporal']]):
                    param.requires_grad = False

        if config.get('freeze_text_decoder', True):
            for name, param in self.text_decoder.named_parameters():
                param.requires_grad = False
        

        self.vision_width = visual_cfg['embed_dim']
        self.text_width = text_cfg.hidden_size

        # self.learnable_token = config.get('learnable_token', True)
        self.learnable_token = True
        if self.learnable_token:
            self.num_learnable_token = config.get('num_learnable_token', 256)
            self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_token, self.vision_width))
            trunc_normal_(self.learnable_queries, std=0.015)

            self.attn_pool = AttentionPool(
                self.vision_width, num_heads=visual_cfg['num_heads'], 
                mlp_ratio=visual_cfg['mlp_ratio'], norm_layer=partial(LayerNormWithForceFP32, eps=1e-6)
            )

        self.visual_fc = nn.Linear(self.vision_width, self.text_width)
        if visual_cfg.get('connect_ln', False):
            self.visual_norm = partial(LayerNormWithForceFP32, eps=1e-6)(self.text_width)
        else:
            self.visual_norm = nn.Identity()
        trunc_normal_(self.visual_fc.weight, std=0.015)

        self.prompt = config.get('prompt', "")
        # prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self.use_contrastive = config.get('use_contrastive', False)
        if self.use_contrastive:
            embed_dim = config.get('contrastive_embed_dim', 256)
            self.vision_proj = nn.Linear(self.vision_width, embed_dim)
            self.text_proj = nn.Linear(self.text_width, embed_dim)
            self.temp = nn.Parameter(torch.ones([]) * config.get('temp', 0.07))


    def forward(self, image, text):
        _, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        query_features = self.visual_norm(self.visual_fc(image_query))
        query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        # targets = text.input_ids.masked_fill(
        #     text.input_ids == self.tokenizer.pad_token_id, 100
        # )
        targets = text.input_ids[:, 1:].clone()
        targets = torch.cat([targets, targets[:, 0:1]], dim=1) # last column is not used

        text_loss_atts = text.attention_mask[:, 1:]
        if self.prompt != "":
            # targets[:, : self.prompt_length] = 100  # do not apply loss to the prompt
            text_loss_atts[:, :self.prompt_length] = 0

        empty_targets = (
            torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        input_embeds = self.text_decoder.dist_model.language_model.embedding.word_embeddings(text.input_ids)
        input_embeds = torch.cat([query_features, input_embeds], dim=1)
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

        loss_mask = torch.cat([1-query_atts, text_loss_atts], dim=1)
        
        outputs = self.text_decoder(
            input_embeds=input_embeds,
            loss_mask=loss_mask,
            labels=targets,
        )
        loss_caption = outputs.loss

        if self.use_contrastive:
            targets_dep = text.input_ids[:, 1:].clone()
            targets_dep = torch.cat([targets_dep, targets_dep[:, 0:1]], dim=1) # last column is not used
            loss_mask_dep = text.attention_mask[:, 1:].clone()
            outputs_text = self.text_decoder(
                tokens=text.input_ids,
                loss_mask=loss_mask_dep,
                labels=targets_dep,
            )

            vision_feats = F.normalize(self.vision_proj(image_query), dim=-1)
            pooled_text_feats = outputs_text.last_hidden_state
            pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), text.attention_mask.sum(dim=-1)-1]
            text_feat = F.normalize(self.text_proj(pooled_text_feats), dim=-1)

            vision_feats_all = torch.cat(all_gather(vision_feats), dim=0)
            text_feat_all = torch.cat(all_gather(text_feat), dim=0)

            sim_q2t = torch.matmul(
                vision_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            sim_i2t = sim_i2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), vision_feats_all.permute(0, 2, 1)
            ).squeeze()

            # text-image similarity: aggregate across all query tokens
            sim_t2i, _ = sim_t2q.max(-1)
            sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

            rank = torch.distributed.get_rank()
            bs = image.size(0)
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(image.device)

            try:
                loss_contrastive = (
                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                ) / 2
            except:
                loss_contrastive = (
                    F.cross_entropy(sim_i2t, targets)
                    + F.cross_entropy(sim_t2i, targets)
                ) / 2
        else:
            loss_contrastive = torch.tensor(0.0).to(image.device)

        return loss_caption, loss_contrastive


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}



class DistributedGPT3_Pretrain_Image(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
        text_cfg = GPT3Config.from_json_file(config['text_cfg'])

        if not config.get('use_eva_g', False):
            self.visual_encoder = VisionTransformer(
                img_size=visual_cfg['img_size'],
                patch_size=visual_cfg['patch_size'], 
                embed_dim=visual_cfg['embed_dim'], depth=visual_cfg['depth'], 
                num_heads=visual_cfg['num_heads'],
                mlp_ratio=visual_cfg['mlp_ratio'], qkv_bias=True,
                norm_layer=partial(LayerNormWithForceFP32, eps=1e-6),
                init_std=0.015, grad_ckpt=visual_cfg.get('grad_ckpt', True),
                drop_path_rate=visual_cfg.get('drop_path', False),
                stop_grad_conv1=visual_cfg.get('stop_grad_conv1', False),
                use_shared_rel_pos_bias=visual_cfg.get('use_shared_rel_pos_bias', False),
                use_abs_pos_emb=visual_cfg.get('use_abs_pos_emb', True),
                init_values=visual_cfg.get('layer_scale_init_value', 0),
                postnorm=visual_cfg.get('postnorm', False),
                clip_model=visual_cfg.get('clip_model', False)
            )
        else:
           self.visual_encoder = create_eva_vit_g(
                img_size=visual_cfg['img_size'],
                norm_layer=partial(LayerNormWithForceFP32, eps=1e-6),
                drop_path_rate=visual_cfg.get('drop_path', False),
                use_checkpoint=True
            )

        pretrained_vit = visual_cfg.get("pretrained_ckpt", None) # timm/xxxxxx
        if pretrained_vit is not None:
            if pretrained_vit.startswith("timm"):
                pretrained_vit_name = pretrained_vit.split("/")[-1]
                pt_weights = timm.create_model(
                    pretrained_vit_name,
                    pretrained=True,
                ).state_dict()
                pt_weights = _convert_pretrained_vit(pt_weights)
            elif pretrained_vit.startswith("clip"):
                pretrained_vit_name = "/".join(pretrained_vit.split("/")[1:])
                pt_weights = torch.load(pretrained_vit_name, map_location='cpu')
                pt_weights = _convert_pretrained_vit(pt_weights)
            elif pretrained_vit.startswith("eva"):
                pretrained_vit_name = "/".join(pretrained_vit.split("/")[1:])
                pt_weights = torch.load(pretrained_vit_name, map_location='cpu')
                interpolate_pos_embed(self.visual_encoder, pt_weights)

            msg = self.visual_encoder.load_state_dict(pt_weights, strict=False)
            print("Initialize Vision Encoder from CKPT {}".format(pretrained_vit_name))
            print(msg)

        
        megatron_cfg = config['megatron_cfg']
        self.text_decoder = DistributedGPT3(
            model_dir=config['text_decoder'],
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            path_load_tag='model', 
            megatron_cfg=megatron_cfg, 
            checkpoint_model_parallel_size=1 if text_cfg.num_hidden_layers < 40 else 8
        )

        if config.get('freeze_vit', False):
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False

        if config.get('freeze_text_decoder', True):
            for name, param in self.text_decoder.named_parameters():
                param.requires_grad = False
        

        self.vision_width = visual_cfg['embed_dim']
        self.text_width = text_cfg.hidden_size

        # self.learnable_token = config.get('learnable_token', True)
        self.learnable_token = True
        if self.learnable_token:
            self.num_learnable_token = config.get('num_learnable_token', 256)
            self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_token, self.vision_width))
            trunc_normal_(self.learnable_queries, std=0.015)

            self.attn_pool = AttentionPool(
                self.vision_width, num_heads=visual_cfg['num_heads'], 
                mlp_ratio=visual_cfg['mlp_ratio'], norm_layer=partial(LayerNormWithForceFP32, eps=1e-6)
            )

        self.visual_fc = nn.Linear(self.vision_width, self.text_width)
        if visual_cfg.get('connect_ln', False):
            self.visual_norm = partial(LayerNormWithForceFP32, eps=1e-6)(self.text_width)
        else:
            self.visual_norm = nn.Identity()
        trunc_normal_(self.visual_fc.weight, std=0.015)

        self.prompt = config.get('prompt', "")
        # prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self.use_contrastive = config.get('use_contrastive', False)
        if self.use_contrastive:
            embed_dim = config.get('contrastive_embed_dim', 256)
            self.vision_proj = nn.Linear(self.vision_width, embed_dim)
            self.text_proj = nn.Linear(self.text_width, embed_dim)
            self.temp = nn.Parameter(torch.ones([]) * config.get('temp', 0.07))


    def forward(self, image, text):
        _, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        query_features = self.visual_norm(self.visual_fc(image_query))
        query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        # targets = text.input_ids.masked_fill(
        #     text.input_ids == self.tokenizer.pad_token_id, 100
        # )

        targets = text.input_ids[:, 1:].clone()
        targets = torch.cat([targets, targets[:, 0:1]], dim=1) # last column is not used
        
        text_loss_atts = text.attention_mask[:, 1:].clone()
        if hasattr(text, "prompt_lengths"):
            prompt_lengths = text.prompt_lengths.cpu().tolist()
            for idx, prompt_length in enumerate(prompt_lengths):
                text_loss_atts[idx, :prompt_length] = 0

        empty_targets = (
            torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        input_embeds = self.text_decoder.dist_model.language_model.embedding.word_embeddings(text.input_ids)
        input_embeds = torch.cat([query_features, input_embeds], dim=1)
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

        loss_mask = torch.cat([1-query_atts, text_loss_atts], dim=1)
        
        outputs = self.text_decoder(
            input_embeds=input_embeds,
            loss_mask=loss_mask,
            labels=targets,
        )
        loss_caption = outputs.loss

        if self.use_contrastive:
            vision_feats = F.normalize(self.vision_proj(image_query), dim=-1)
            pooled_text_feats = outputs.last_hidden_state
            pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), text.attention_mask.sum(dim=-1)-1]
            text_feat = F.normalize(self.text_proj(pooled_text_feats), dim=-1)

            vision_feats_all = torch.cat(all_gather(vision_feats), dim=0)
            text_feat_all = torch.cat(all_gather(text_feat), dim=0)

            sim_q2t = torch.matmul(
                vision_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
            ).squeeze()
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            sim_i2t = sim_i2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), vision_feats_all.permute(0, 2, 1)
            ).squeeze()

            # text-image similarity: aggregate across all query tokens
            sim_t2i, _ = sim_t2q.max(-1)
            sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

            rank = torch.distributed.get_rank()
            bs = image.size(0)
            targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(image.device)

            try:
                loss_contrastive = (
                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                ) / 2
            except:
                loss_contrastive = (
                    F.cross_entropy(sim_i2t, targets)
                    + F.cross_entropy(sim_t2i, targets)
                ) / 2
        else:
            loss_contrastive = torch.tensor(0.0).to(image.device)

        return loss_caption, loss_contrastive


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token'}



class DistributedGPT3_Cls(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
        text_cfg = GPT3Config.from_json_file(config['text_cfg'])

        self.visual_encoder = TimeSformer(
            img_size=visual_cfg['img_size'], num_frames=config['num_frames'],
            patch_size=visual_cfg['patch_size'], 
            embed_dim=visual_cfg['embed_dim'], depth=visual_cfg['depth'], 
            num_heads=visual_cfg['num_heads'],
            mlp_ratio=visual_cfg['mlp_ratio'], qkv_bias=True,
            norm_layer=partial(LayerNormWithForceFP32, eps=1e-6),
            init_std=0.015, grad_ckpt=visual_cfg.get('grad_ckpt', True),
            drop_path_rate=visual_cfg.get('drop_path', False),
            stop_grad_conv1=visual_cfg.get('stop_grad_conv1', False),
            use_shared_rel_pos_bias=visual_cfg.get('use_shared_rel_pos_bias', False),
            use_abs_pos_emb=visual_cfg.get('use_abs_pos_emb', True),
            init_values=visual_cfg.get('layer_scale_init_value', 0),
            postnorm=visual_cfg.get('postnorm', False),
            clip_model=visual_cfg.get('clip_model', False)
        )

        pretrained_vit = visual_cfg.get("pretrained_ckpt", None) # timm/xxxxxx
        if pretrained_vit is not None:
            if pretrained_vit.startswith("timm"):
                pretrained_vit_name = pretrained_vit.split("/")[-1]
                pt_weights = timm.create_model(
                    pretrained_vit_name,
                    pretrained=True,
                ).state_dict()
                pt_weights = _convert_pretrained_vit(pt_weights)
            elif pretrained_vit.startswith("clip"):
                pretrained_vit_name = "/".join(pretrained_vit.split("/")[1:])
                pt_weights = torch.load(pretrained_vit_name, map_location='cpu')
                pt_weights = _convert_pretrained_vit(pt_weights)

            msg = self.visual_encoder.load_state_dict(pt_weights, strict=False)
            print("Initialize Vision Encoder from CKPT {}".format(pretrained_vit_name))
            print(msg)

        
        megatron_cfg = config['megatron_cfg']
        # megatron_cfg['world_size'] = torch.distributed.get_world_size()
        # megatron_cfg['world_size'] = 1
        self.text_decoder = DistributedGPT3(
            model_dir=config['text_decoder'],
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            path_load_tag='model', 
            megatron_cfg=megatron_cfg, 
            # load_state_dict=False,
            checkpoint_model_parallel_size=1 if text_cfg.num_hidden_layers < 40 else 8
        )

        if config.get('freeze_vit', False):
            for name, param in self.visual_encoder.named_parameters():
                if not any([x in name for x in ['time', 'temporal']]):
                    param.requires_grad = False

        if config.get('freeze_text_decoder', True):
            for name, param in self.text_decoder.named_parameters():
                param.requires_grad = False
        

        self.vision_width = visual_cfg['embed_dim']
        self.text_width = text_cfg.hidden_size

        # self.learnable_token = config.get('learnable_token', True)
        self.learnable_token = True
        if self.learnable_token:
            self.num_learnable_token = config.get('num_learnable_token', 256)
            self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_token, self.vision_width))
            trunc_normal_(self.learnable_queries, std=0.015)

            self.attn_pool = AttentionPool(
                self.vision_width, num_heads=visual_cfg['num_heads'], 
                mlp_ratio=visual_cfg['mlp_ratio'], norm_layer=partial(LayerNormWithForceFP32, eps=1e-6)
            )

        self.visual_fc = nn.Linear(self.vision_width, self.text_width)
        if visual_cfg.get('connect_ln', False):
            self.visual_norm = partial(LayerNormWithForceFP32, eps=1e-6)(self.text_width)
        else:
            self.visual_norm = nn.Identity()
        trunc_normal_(self.visual_fc.weight, std=0.015)

        self.prompt = config.get('prompt', "")
        # prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self.use_cls = config.get('use_cls', False)
        if self.use_cls:
            self.cls_head = nn.Sequential(
                nn.Linear(self.text_width, self.text_width),
                nn.ReLU(),
                nn.Linear(self.text_width, config['num_classes'])
            )

    def forward(self, image, text=None, prompt_text=None, labels=None, train=True):
        _, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        query_features = self.visual_norm(self.visual_fc(image_query))
        query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        prompt_lengths = text.prompt_lengths.cpu().tolist()

        targets = text.input_ids[:, 1:].clone()
        targets = torch.cat([targets, targets[:, 0:1]], dim=1) # last column is not used
        
        text_loss_atts = text.attention_mask[:, 1:].clone()
        for idx, prompt_length in enumerate(prompt_lengths):
            text_loss_atts[idx, :prompt_length] = 0

        if train:
            empty_targets = (
                torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            input_embeds = self.text_decoder.dist_model.language_model.embedding.word_embeddings(text.input_ids)
            input_embeds = torch.cat([query_features, input_embeds], dim=1)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            loss_mask = torch.cat([1-query_atts, text_loss_atts], dim=1)
            
            outputs = self.text_decoder(
                input_embeds=input_embeds,
                loss_mask=loss_mask,
                labels=targets,
            )

            loss_caption = outputs.loss

            if self.use_cls:
                targets_tmp = prompt_text.input_ids[:, 1:].clone()
                targets_tmp = torch.cat([targets_tmp, targets_tmp[:, 0:1]], dim=1) # last column is not used
                empty_targets_tmp = torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
                targets_tmp = torch.cat([empty_targets_tmp, targets_tmp], dim=1)

                input_embeds_prompt = self.text_decoder.dist_model.language_model.embedding.word_embeddings(prompt_text.input_ids)
                input_embeds_prompt = torch.cat([query_features, input_embeds_prompt], dim=1)
                attention_mask_prompt = torch.cat([query_atts, prompt_text.attention_mask], dim=1)

                text_loss_atts_prompt = 1-prompt_text.attention_mask[:, 1:].clone()
                loss_mask_prompt = torch.cat([1-query_atts, text_loss_atts_prompt], dim=1)
                
                outputs_cls = self.text_decoder(
                    input_embeds=input_embeds_prompt,
                    loss_mask=loss_mask_prompt,
                    labels=targets_tmp,
                )

                pooled_text_feats = outputs_cls.last_hidden_state
                pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), attention_mask_prompt.sum(dim=-1)-1]
                logits = self.cls_head(pooled_text_feats)

                loss_cls = F.cross_entropy(logits, labels)
            else:
                loss_cls = torch.tensor(0.0).to(image.device)

            return loss_caption, loss_cls

        else:
            num_cls = text.input_ids.shape[0] // image.shape[0]
            query_features_per_video = query_features.clone()
            query_features = query_features.unsqueeze(1).repeat(1, num_cls, 1, 1)
            query_features = rearrange(query_features, 'b q n c -> (b q) n c')

            query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)
            
            empty_targets = (
                torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            input_embeds = self.text_decoder.dist_model.language_model.embedding.word_embeddings(text.input_ids)
            input_embeds = torch.cat([query_features, input_embeds], dim=1)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            loss_mask = torch.cat([1-query_atts, text_loss_atts], dim=1)
            
            outputs = self.text_decoder(
                input_embeds=input_embeds,
                loss_mask=loss_mask,
                labels=targets,
            )
            
            generation_logits = -torch.sum(outputs.losses * loss_mask, dim=-1)
            generation_logits = rearrange(generation_logits, '(b c) -> b c', c=num_cls)
            generation_logits = torch.softmax(generation_logits, dim=-1)

            if self.use_cls:
                query_atts_per_video = torch.ones(query_features_per_video.size()[:-1], dtype=torch.long).to(image.device)
                targets_tmp = prompt_text.input_ids[:, 1:].clone()
                targets_tmp = torch.cat([targets_tmp, targets_tmp[:, 0:1]], dim=1) # last column is not used
                empty_targets_tmp = torch.ones(query_atts_per_video.size(), dtype=torch.long).to(image.device).fill_(100)
                targets_tmp = torch.cat([empty_targets_tmp, targets_tmp], dim=1)

                input_embeds_prompt = self.text_decoder.dist_model.language_model.embedding.word_embeddings(prompt_text.input_ids)
                input_embeds_prompt = torch.cat([query_features_per_video, input_embeds_prompt], dim=1)
                attention_mask_prompt = torch.cat([query_atts_per_video, prompt_text.attention_mask], dim=1)

                text_loss_atts_prompt = prompt_text.attention_mask[:, 1:].clone()
                loss_mask_prompt = torch.cat([1-query_atts_per_video, text_loss_atts_prompt], dim=1)
                
                outputs_cls = self.text_decoder(
                    input_embeds=input_embeds_prompt,
                    loss_mask=loss_mask_prompt,
                    labels=targets_tmp,
                )

                pooled_text_feats = outputs_cls.last_hidden_state
                pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), attention_mask_prompt.sum(dim=-1)-1]
                cls_logits = self.cls_head(pooled_text_feats)
            else:
                cls_logits = None

            return generation_logits, cls_logits

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}




class DistributedGPT3_Caption(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
        text_cfg = GPT3Config.from_json_file(config['text_cfg'])

        self.visual_encoder = TimeSformer(
            img_size=visual_cfg['img_size'], num_frames=config['num_frames'],
            patch_size=visual_cfg['patch_size'], 
            embed_dim=visual_cfg['embed_dim'], depth=visual_cfg['depth'], 
            num_heads=visual_cfg['num_heads'],
            mlp_ratio=visual_cfg['mlp_ratio'], qkv_bias=True,
            norm_layer=partial(LayerNormWithForceFP32, eps=1e-6),
            init_std=0.015, grad_ckpt=visual_cfg.get('grad_ckpt', True),
            drop_path_rate=visual_cfg.get('drop_path', False),
            stop_grad_conv1=visual_cfg.get('stop_grad_conv1', False),
            use_shared_rel_pos_bias=visual_cfg.get('use_shared_rel_pos_bias', False),
            use_abs_pos_emb=visual_cfg.get('use_abs_pos_emb', True),
            init_values=visual_cfg.get('layer_scale_init_value', 0),
            postnorm=visual_cfg.get('postnorm', False),
            clip_model=visual_cfg.get('clip_model', False)
        )

        pretrained_vit = visual_cfg.get("pretrained_ckpt", None) # timm/xxxxxx
        if pretrained_vit is not None:
            if pretrained_vit.startswith("timm"):
                pretrained_vit_name = pretrained_vit.split("/")[-1]
                pt_weights = timm.create_model(
                    pretrained_vit_name,
                    pretrained=True,
                ).state_dict()
                pt_weights = _convert_pretrained_vit(pt_weights)
            elif pretrained_vit.startswith("clip"):
                pretrained_vit_name = "/".join(pretrained_vit.split("/")[1:])
                pt_weights = torch.load(pretrained_vit_name, map_location='cpu')
                pt_weights = _convert_pretrained_vit(pt_weights)

            msg = self.visual_encoder.load_state_dict(pt_weights, strict=False)
            print("Initialize Vision Encoder from CKPT {}".format(pretrained_vit_name))
            print(msg)

        
        megatron_cfg = config['megatron_cfg']
        # megatron_cfg['world_size'] = torch.distributed.get_world_size()
        # megatron_cfg['world_size'] = 1
        self.text_decoder = DistributedGPT3(
            model_dir=config['text_decoder'],
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            path_load_tag='model', 
            megatron_cfg=megatron_cfg, 
            checkpoint_model_parallel_size=1 if text_cfg.num_hidden_layers < 40 else 8
        )

        if config.get('freeze_vit', False):
            for name, param in self.visual_encoder.named_parameters():
                if not any([x in name for x in ['time', 'temporal']]):
                    param.requires_grad = False

        if config.get('freeze_text_decoder', True):
            for name, param in self.text_decoder.named_parameters():
                param.requires_grad = False
        

        self.vision_width = visual_cfg['embed_dim']
        self.text_width = text_cfg.hidden_size

        # self.learnable_token = config.get('learnable_token', True)
        self.learnable_token = True
        if self.learnable_token:
            self.num_learnable_token = config.get('num_learnable_token', 256)
            self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_token, self.vision_width))
            trunc_normal_(self.learnable_queries, std=0.015)

            self.attn_pool = AttentionPool(
                self.vision_width, num_heads=visual_cfg['num_heads'], 
                mlp_ratio=visual_cfg['mlp_ratio'], norm_layer=partial(LayerNormWithForceFP32, eps=1e-6)
            )

        self.visual_fc = nn.Linear(self.vision_width, self.text_width)
        if visual_cfg.get('connect_ln', False):
            self.visual_norm = partial(LayerNormWithForceFP32, eps=1e-6)(self.text_width)
        else:
            self.visual_norm = nn.Identity()
        trunc_normal_(self.visual_fc.weight, std=0.015)

        self.prompt = config.get('prompt', "")
        
    def forward(self, image, text=None):
        _, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        query_features = self.visual_norm(self.visual_fc(image_query))
        query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        prompt_lengths = text.prompt_lengths.cpu().tolist()

        targets = text.input_ids[:, 1:].clone()
        targets = torch.cat([targets, targets[:, 0:1]], dim=1) # last column is not used
        
        text_loss_atts = text.attention_mask[:, 1:].clone()
        for idx, prompt_length in enumerate(prompt_lengths):
            text_loss_atts[idx, :prompt_length] = 0

        empty_targets = (
            torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        input_embeds = self.text_decoder.dist_model.language_model.embedding.word_embeddings(text.input_ids)
        input_embeds = torch.cat([query_features, input_embeds], dim=1)
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

        loss_mask = torch.cat([1-query_atts, text_loss_atts], dim=1)
        
        outputs = self.text_decoder(
            input_embeds=input_embeds,
            loss_mask=loss_mask,
            labels=targets,
        )

        loss_caption = outputs.loss

        return loss_caption

    def generate(self, image, text):
        _, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        query_features = self.visual_norm(self.visual_fc(image_query))
        query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        res = []
        for i in range(len(text.input_ids)):
            output = self.text_decoder.generate(
                text.input_ids[i:i+1], 
                query_embeds=query_features[i:i+1],
                termination_id=self.tokenizer.tokenizer.eos,
                do_sample=False,
                prompt_length=text.attention_mask.sum(-1)[i]-1
            )
            res.append(output.sequences.cpu())
        return res


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}


class DistributedGPT3_Retrieval(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
        text_cfg = GPT3Config.from_json_file(config['text_cfg'])

        self.visual_encoder = TimeSformer(
            img_size=visual_cfg['img_size'], num_frames=config['num_frames'],
            patch_size=visual_cfg['patch_size'], 
            embed_dim=visual_cfg['embed_dim'], depth=visual_cfg['depth'], 
            num_heads=visual_cfg['num_heads'],
            mlp_ratio=visual_cfg['mlp_ratio'], qkv_bias=True,
            norm_layer=partial(LayerNormWithForceFP32, eps=1e-6),
            init_std=0.015, grad_ckpt=visual_cfg.get('grad_ckpt', True),
            drop_path_rate=visual_cfg.get('drop_path', False),
            stop_grad_conv1=visual_cfg.get('stop_grad_conv1', False),
            use_shared_rel_pos_bias=visual_cfg.get('use_shared_rel_pos_bias', False),
            use_abs_pos_emb=visual_cfg.get('use_abs_pos_emb', True),
            init_values=visual_cfg.get('layer_scale_init_value', 0),
            postnorm=visual_cfg.get('postnorm', False),
            clip_model=visual_cfg.get('clip_model', False)
        )

        pretrained_vit = visual_cfg.get("pretrained_ckpt", None) # timm/xxxxxx
        if pretrained_vit is not None:
            if pretrained_vit.startswith("timm"):
                pretrained_vit_name = pretrained_vit.split("/")[-1]
                pt_weights = timm.create_model(
                    pretrained_vit_name,
                    pretrained=True,
                ).state_dict()
                pt_weights = _convert_pretrained_vit(pt_weights)
            elif pretrained_vit.startswith("clip"):
                pretrained_vit_name = "/".join(pretrained_vit.split("/")[1:])
                pt_weights = torch.load(pretrained_vit_name, map_location='cpu')
                pt_weights = _convert_pretrained_vit(pt_weights)

            msg = self.visual_encoder.load_state_dict(pt_weights, strict=False)
            print("Initialize Vision Encoder from CKPT {}".format(pretrained_vit_name))
            print(msg)

        
        megatron_cfg = config['megatron_cfg']
        # megatron_cfg['world_size'] = torch.distributed.get_world_size()
        # megatron_cfg['world_size'] = 1
        self.text_decoder = DistributedGPT3(
            model_dir=config['text_decoder'],
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            path_load_tag='model', 
            megatron_cfg=megatron_cfg, 
            checkpoint_model_parallel_size=1 if text_cfg.num_hidden_layers < 40 else 8
        )

        if config.get('freeze_vit', False):
            for name, param in self.visual_encoder.named_parameters():
                if not any([x in name for x in ['time', 'temporal']]):
                    param.requires_grad = False

        if config.get('freeze_text_decoder', True):
            for name, param in self.text_decoder.named_parameters():
                param.requires_grad = False
        

        self.vision_width = visual_cfg['embed_dim']
        self.text_width = text_cfg.hidden_size

        # self.learnable_token = config.get('learnable_token', True)
        self.learnable_token = True
        if self.learnable_token:
            self.num_learnable_token = config.get('num_learnable_token', 256)
            self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_token, self.vision_width))
            trunc_normal_(self.learnable_queries, std=0.015)

            self.attn_pool = AttentionPool(
                self.vision_width, num_heads=visual_cfg['num_heads'], 
                mlp_ratio=visual_cfg['mlp_ratio'], norm_layer=partial(LayerNormWithForceFP32, eps=1e-6)
            )

        self.visual_fc = nn.Linear(self.vision_width, self.text_width)
        if visual_cfg.get('connect_ln', False):
            self.visual_norm = partial(LayerNormWithForceFP32, eps=1e-6)(self.text_width)
        else:
            self.visual_norm = nn.Identity()
        trunc_normal_(self.visual_fc.weight, std=0.015)
        
        embed_dim = config.get('contrastive_embed_dim', 256)
        self.vision_proj = nn.Linear(self.vision_width, embed_dim)
        self.text_proj = nn.Linear(self.text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * config.get('temp', 0.07))
    
    def extract_vision_feature(self, image):
        image_query, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        # query_features = self.visual_norm(self.visual_fc(image_query))
        # query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        vision_feats = F.normalize(self.vision_proj(image_query), dim=-1)
        return vision_feats
    

    def extract_text_feature(self, text):
        targets = text.input_ids[:, 1:].clone()
        targets = torch.cat([targets, targets[:, 0:1]], dim=1) # last column is not used
        loss_mask = text.attention_mask[:, 1:].clone()
        outputs = self.text_decoder(
            tokens=text.input_ids,
            loss_mask=loss_mask,
            labels=targets,
        )

        pooled_text_feats = outputs.last_hidden_state
        pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), text.attention_mask.sum(dim=-1)-1]
        text_feat = F.normalize(self.text_proj(pooled_text_feats), dim=-1)
        return text_feat


    def forward(self, image, text, idx):
        image_query, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        # query_features = self.visual_norm(self.visual_fc(image_query))
        # query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        vision_feats = F.normalize(self.vision_proj(image_query), dim=-1)

        targets = text.input_ids[:, 1:].clone()
        targets = torch.cat([targets, targets[:, 0:1]], dim=1) # last column is not used
        loss_mask = text.attention_mask[:, 1:].clone()
        outputs = self.text_decoder(
            tokens=text.input_ids,
            loss_mask=loss_mask,
            labels=targets,
        )

        pooled_text_feats = outputs.last_hidden_state
        pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), text.attention_mask.sum(dim=-1)-1]
        text_feat = F.normalize(self.text_proj(pooled_text_feats), dim=-1)

        vision_feats_all = torch.cat(all_gather(vision_feats), dim=0)
        text_feat_all = torch.cat(all_gather(text_feat), dim=0)
        gathered_idx = concat_all_gather(idx).view(-1, 1)

        sim_i2t = torch.matmul(vision_feats, text_feat_all.permute(1, 0)) / self.temp
        sim_t2i = torch.matmul(text_feat, vision_feats_all.permute(1, 0)) / self.temp


        idx = idx.view(-1, 1)
        pos_idx = torch.eq(idx, gathered_idx.t()).float()
        targets = pos_idx / pos_idx.sum(1, keepdim=True)


        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * targets, dim=1).mean()

        loss_contrastive = (loss_i2t + loss_t2i) / 2

        return loss_contrastive


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}


class DistributedGPT3_Retrieval_Cls(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
        text_cfg = GPT3Config.from_json_file(config['text_cfg'])

        self.visual_encoder = TimeSformer(
            img_size=visual_cfg['img_size'], num_frames=config['num_frames'],
            patch_size=visual_cfg['patch_size'], 
            embed_dim=visual_cfg['embed_dim'], depth=visual_cfg['depth'], 
            num_heads=visual_cfg['num_heads'],
            mlp_ratio=visual_cfg['mlp_ratio'], qkv_bias=True,
            norm_layer=partial(LayerNormWithForceFP32, eps=1e-6),
            init_std=0.015, grad_ckpt=visual_cfg.get('grad_ckpt', True),
            drop_path_rate=visual_cfg.get('drop_path', False),
            stop_grad_conv1=visual_cfg.get('stop_grad_conv1', False),
            use_shared_rel_pos_bias=visual_cfg.get('use_shared_rel_pos_bias', False),
            use_abs_pos_emb=visual_cfg.get('use_abs_pos_emb', True),
            init_values=visual_cfg.get('layer_scale_init_value', 0),
            postnorm=visual_cfg.get('postnorm', False),
            clip_model=visual_cfg.get('clip_model', False)
        )

        pretrained_vit = visual_cfg.get("pretrained_ckpt", None) # timm/xxxxxx
        if pretrained_vit is not None:
            if pretrained_vit.startswith("timm"):
                pretrained_vit_name = pretrained_vit.split("/")[-1]
                pt_weights = timm.create_model(
                    pretrained_vit_name,
                    pretrained=True,
                ).state_dict()
                pt_weights = _convert_pretrained_vit(pt_weights)
            elif pretrained_vit.startswith("clip"):
                pretrained_vit_name = "/".join(pretrained_vit.split("/")[1:])
                pt_weights = torch.load(pretrained_vit_name, map_location='cpu')
                pt_weights = _convert_pretrained_vit(pt_weights)

            msg = self.visual_encoder.load_state_dict(pt_weights, strict=False)
            print("Initialize Vision Encoder from CKPT {}".format(pretrained_vit_name))
            print(msg)

        
        megatron_cfg = config['megatron_cfg']
        # megatron_cfg['world_size'] = torch.distributed.get_world_size()
        # megatron_cfg['world_size'] = 1
        self.text_decoder = DistributedGPT3(
            model_dir=config['text_decoder'],
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            path_load_tag='model', 
            megatron_cfg=megatron_cfg, 
            # load_state_dict=False,
            checkpoint_model_parallel_size=1 if text_cfg.num_hidden_layers < 40 else 8
        )

        if config.get('freeze_vit', False):
            for name, param in self.visual_encoder.named_parameters():
                if not any([x in name for x in ['time', 'temporal']]):
                    param.requires_grad = False

        if config.get('freeze_text_decoder', True):
            for name, param in self.text_decoder.named_parameters():
                param.requires_grad = False
        

        self.vision_width = visual_cfg['embed_dim']
        self.text_width = text_cfg.hidden_size

        # self.learnable_token = config.get('learnable_token', True)
        self.learnable_token = True
        if self.learnable_token:
            self.num_learnable_token = config.get('num_learnable_token', 256)
            self.learnable_queries = nn.Parameter(torch.randn(1, self.num_learnable_token, self.vision_width))
            trunc_normal_(self.learnable_queries, std=0.015)

            self.attn_pool = AttentionPool(
                self.vision_width, num_heads=visual_cfg['num_heads'], 
                mlp_ratio=visual_cfg['mlp_ratio'], norm_layer=partial(LayerNormWithForceFP32, eps=1e-6)
            )

        self.visual_fc = nn.Linear(self.vision_width, self.text_width)
        if visual_cfg.get('connect_ln', False):
            self.visual_norm = partial(LayerNormWithForceFP32, eps=1e-6)(self.text_width)
        else:
            self.visual_norm = nn.Identity()
        trunc_normal_(self.visual_fc.weight, std=0.015)

        self.prompt = config.get('prompt', "")
        # prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self.use_cls = config.get('use_cls', False)
        if self.use_cls:
            self.cls_head = nn.Sequential(
                nn.Linear(self.text_width, self.text_width),
                nn.ReLU(),
                nn.Linear(self.text_width, config['num_classes'])
            )

    def forward(self, image, text=None, prompt_text=None, negative_indices=None, labels=None, train=True):
        _, image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_query = self.attn_pool(self.learnable_queries.repeat(image_embeds.shape[0], 1, 1), image_embeds)

        query_features = self.visual_norm(self.visual_fc(image_query))
        query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

        prompt_lengths = text.prompt_lengths.cpu().tolist()
        targets = text.input_ids[:, 1:].clone()
        targets = torch.cat([targets, targets[:, 0:1]], dim=1) # last column is not used
        
        text_loss_atts = text.attention_mask[:, 1:].clone()
        for idx, prompt_length in enumerate(prompt_lengths):
            text_loss_atts[idx, :prompt_length] = 0

        if train:            
            query_features = torch.cat([
                query_features, 
                query_features[negative_indices]
            ], dim = 0)
            query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)

            empty_targets = (
                torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            input_embeds = self.text_decoder.dist_model.language_model.embedding.word_embeddings(text.input_ids)
            input_embeds = torch.cat([query_features, input_embeds], dim=1)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            loss_mask = torch.cat([1-query_atts, text_loss_atts], dim=1)
            
            outputs = self.text_decoder(
                input_embeds=input_embeds,
                loss_mask=loss_mask,
                labels=targets,
            )

            loss_caption = outputs.loss

            if self.use_cls:
                targets_tmp = prompt_text.input_ids[:, 1:].clone()
                targets_tmp = torch.cat([targets_tmp, targets_tmp[:, 0:1]], dim=1) # last column is not used
                empty_targets_tmp = torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
                targets_tmp = torch.cat([empty_targets_tmp, targets_tmp], dim=1)

                input_embeds_prompt = self.text_decoder.dist_model.language_model.embedding.word_embeddings(prompt_text.input_ids)
                input_embeds_prompt = torch.cat([query_features, input_embeds_prompt], dim=1)
                attention_mask_prompt = torch.cat([query_atts, prompt_text.attention_mask], dim=1)

                text_loss_atts_prompt = 1-prompt_text.attention_mask[:, 1:].clone()
                loss_mask_prompt = torch.cat([1-query_atts, text_loss_atts_prompt], dim=1)
                
                outputs_cls = self.text_decoder(
                    input_embeds=input_embeds_prompt,
                    loss_mask=loss_mask_prompt,
                    labels=targets_tmp,
                )

                pooled_text_feats = outputs_cls.last_hidden_state
                pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), attention_mask_prompt.sum(dim=-1)-1]
                logits = self.cls_head(pooled_text_feats)

                loss_cls = F.cross_entropy(logits, labels)
            else:
                loss_cls = torch.tensor(0.0).to(image.device)

            return loss_caption, loss_cls

        else:
            t = text.input_ids.shape[0] // query_features.shape[0]
            query_features = repeat(query_features, 'v n c -> (v t) n c', t=t)

            query_features_per_video = query_features.clone()
            query_atts = torch.ones(query_features.size()[:-1], dtype=torch.long).to(image.device)
            
            empty_targets = (
                torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            input_embeds = self.text_decoder.dist_model.language_model.embedding.word_embeddings(text.input_ids)
            input_embeds = torch.cat([query_features, input_embeds], dim=1)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            loss_mask = torch.cat([1-query_atts, text_loss_atts], dim=1)
            
            outputs = self.text_decoder(
                input_embeds=input_embeds,
                loss_mask=loss_mask,
                labels=targets,
            )
            
            generation_logits = -torch.sum(outputs.losses * loss_mask, dim=-1)
            generation_logits = rearrange(generation_logits, '(v t) -> v t', t=t)

            if self.use_cls:
                query_atts_per_video = torch.ones(query_features_per_video.size()[:-1], dtype=torch.long).to(image.device)
                targets_tmp = prompt_text.input_ids[:, 1:].clone()
                targets_tmp = torch.cat([targets_tmp, targets_tmp[:, 0:1]], dim=1) # last column is not used
                empty_targets_tmp = torch.ones(query_atts_per_video.size(), dtype=torch.long).to(image.device).fill_(100)
                targets_tmp = torch.cat([empty_targets_tmp, targets_tmp], dim=1)

                input_embeds_prompt = self.text_decoder.dist_model.language_model.embedding.word_embeddings(prompt_text.input_ids)
                input_embeds_prompt = torch.cat([query_features_per_video, input_embeds_prompt], dim=1)
                attention_mask_prompt = torch.cat([query_atts_per_video, prompt_text.attention_mask], dim=1)

                text_loss_atts_prompt = prompt_text.attention_mask[:, 1:].clone()
                loss_mask_prompt = torch.cat([1-query_atts_per_video, text_loss_atts_prompt], dim=1)
                
                outputs_cls = self.text_decoder(
                    input_embeds=input_embeds_prompt,
                    loss_mask=loss_mask_prompt,
                    labels=targets_tmp,
                )

                pooled_text_feats = outputs_cls.last_hidden_state
                pooled_text_feats = pooled_text_feats[torch.arange(pooled_text_feats.shape[0]), attention_mask_prompt.sum(dim=-1)-1]
                cls_logits = torch.softmax(self.cls_head(pooled_text_feats), dim=-1)[:, 1]
                cls_logits = rearrange(cls_logits, '(v t) -> v t', t=t)
            else:
                cls_logits = None

            return generation_logits, cls_logits

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output