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
from models.modeling_alpro import BertConfig, BertForMaskedLM
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

from einops import rearrange

class ALPRO_Pretrain(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.mlm_probability = config['mlm_probability']
        self.embed_dim = config['embed_dim']
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_encoder)
        self.text_width = self.config_encoder.hidden_size

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
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

        self.large = False
        if self.config_encoder.hidden_size != visual_cfg['embed_dim']:
            self.visn_fc = nn.Linear(visual_cfg['embed_dim'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        print("large: ", self.large)

        self.vision_proj = nn.Linear(self.text_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.itm_head = nn.Linear(self.text_width, 2)


    def forward(self, image, text):
        batch_size, C, T, H, W = image.size()
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        _, image_embeds = self.visual_encoder(image)
        cls_token, image_embeds_ = image_embeds[:, 0:1], image_embeds[:, 1:]
        image_embeds_ = rearrange(image_embeds_, 'b (t n) c -> b t n c', t=T).mean(1)
        image_embeds = torch.cat([cls_token, image_embeds_], dim=1)

        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode = 'text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        gathered_image_feats = torch.cat(all_gather(image_feat), dim=0)
        gathered_text_feats = torch.cat(all_gather(text_feat), dim=0)

        sim_i2t = image_feat @ gathered_text_feats.t() / self.temp 
        sim_t2i = text_feat @ gathered_image_feats.t() / self.temp 
                             
        # [IMPORTANT] be very careful when initializing the GT sim_v2t 
        # allgather return the concatenated features in the order of local_rank()
        sim_targets = torch.zeros_like(sim_i2t)

        b, local_rank = image.shape[0], torch.distributed.get_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.eye(b)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t + loss_t2i) / 2

        ###=================================###
        # forward the positve image-text pair
        attention_mask = torch.cat([text.attention_mask, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        attention_mask_all = torch.cat([text_atts_all, image_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, image_embeds_all], dim=1)

        encoder_outputs_neg = self.text_encoder.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= MLM ========================##
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix=probability_matrix)

        text_output = self.text_encoder.bert(input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state

        attention_mask = torch.cat([text.attention_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )

        txt_len = text.attention_mask.shape[1]
        txt_output = encoder_outputs.last_hidden_state[:, :txt_len]

        mlm_logits = self.text_encoder.cls(txt_output)

        loss_fct = nn.CrossEntropyLoss()
        loss_mlm = loss_fct(mlm_logits.view(-1, self.config_encoder.vocab_size), labels.view(-1))

        return loss_mlm, loss_ita, loss_itm

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}


class ALPRO_Retrieval(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.embed_dim = config['embed_dim']
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_encoder)
        self.text_width = self.config_encoder.hidden_size

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
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

        self.large = False
        if self.config_encoder.hidden_size != visual_cfg['embed_dim']:
            self.visn_fc = nn.Linear(visual_cfg['embed_dim'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        print("large: ", self.large)

        self.vision_proj = nn.Linear(self.text_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.itm_head = nn.Linear(self.text_width, 2)


    def forward(self, image, text, idx):
        batch_size, C, T, H, W = image.size()
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        _, image_embeds = self.visual_encoder(image)
        cls_token, image_embeds_ = image_embeds[:, 0:1], image_embeds[:, 1:]
        image_embeds_ = rearrange(image_embeds_, 'b (t n) c -> b t n c', t=T).mean(1)
        image_embeds = torch.cat([cls_token, image_embeds_], dim=1)

        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode = 'text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        gathered_image_feats = torch.cat(all_gather(image_feat), dim=0)
        gathered_text_feats = torch.cat(all_gather(text_feat), dim=0)
        gathered_idx = concat_all_gather(idx).view(-1, 1)

        sim_i2t = image_feat @ gathered_text_feats.t() / self.temp 
        sim_t2i = text_feat @ gathered_image_feats.t() / self.temp 
                             
        # [IMPORTANT] be very careful when initializing the GT sim_v2t 
        # allgather return the concatenated features in the order of local_rank()
        # sim_targets = torch.zeros_like(sim_i2t)

        # b, local_rank = image.shape[0], torch.distributed.get_rank()
        # b_start, b_end = b * local_rank, b * (local_rank + 1)
        # sim_targets[:, b_start: b_end] = torch.eye(b)

        idx = idx.view(-1, 1)
        pos_idx = torch.eq(idx, gathered_idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t + loss_t2i) / 2

        ###=================================###
        # forward the positve image-text pair
        attention_mask = torch.cat([text.attention_mask, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        attention_mask_all = torch.cat([text_atts_all, image_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, image_embeds_all], dim=1)

        encoder_outputs_neg = self.text_encoder.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}


class ALPRO_Cls(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer
        
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_encoder)
        self.text_width = self.config_encoder.hidden_size

        visual_cfg = json.load(open(config['visual_cfg'], 'r'))
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

        self.large = False
        if self.config_encoder.hidden_size != visual_cfg['embed_dim']:
            self.visn_fc = nn.Linear(visual_cfg['embed_dim'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        print("large: ", self.large)

        self.cls_head = nn.Sequential(
            nn.Linear(self.text_width, self.text_width),
            nn.ReLU(),
            nn.Linear(self.text_width, config['num_classes'])
        )

    def forward(self, image, text, labels=None):
        batch_size, C, T, H, W = image.size()
        _, image_embeds = self.visual_encoder(image)
        cls_token, image_embeds_ = image_embeds[:, 0:1], image_embeds[:, 1:]
        image_embeds_ = rearrange(image_embeds_, 'b (t n) c -> b t n c', t=T).mean(1)
        image_embeds = torch.cat([cls_token, image_embeds_], dim=1)

        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode = 'text')
        text_embeds = text_output.last_hidden_state

        ###=================================###
        # forward the positve image-text pair
        attention_mask = torch.cat([text.attention_mask, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        logits = self.cls_head(encoder_outputs.last_hidden_state[:,0,:])

        if labels is None:
            return logits
        else:
            loss = F.cross_entropy(logits, labels)
            return logits, loss

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