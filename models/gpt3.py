'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vision_transformer import (
    TimeSformer, LayerNormWithForceFP32, 
    _convert_pretrained_vit, AttentionPool
)
from models.modeling_gpt3 import GPT3Config, GPT3Model
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


class GPT3_Pretrain(nn.Module):
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

        
        self.text_decoder = GPT3Model.from_pretrained(config['text_decoder'])

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
        prompt_tokens = self.tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        
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

        targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )

        if self.prompt != "":
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(query_atts.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        input_embeds = self.text_decoder.language_model.word_embeddings(text.input_ids)
        input_embeds = torch.cat([query_features, input_embeds], dim=1)
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
        
        outputs = self.text_decoder(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
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