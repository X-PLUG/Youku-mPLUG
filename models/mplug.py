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
from models.modeling_mplug import BertConfig, FusionForMaskedLM, BertPrefixModel, BertModel
from models.distributed_utils import all_gather

import torch
import torch.nn.functional as F
from torch import nn

import timm
from timm.models.layers import trunc_normal_
from models.predictor_mplug import TextGenerator

import json
import numpy as np
import random
import math


class mPLUG_Pretrain(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.mlm_probability = config['mlm_probability']
        self.embed_dim = config['embed_dim']
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.text_width = self.config_encoder.hidden_size
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decoder_layers

        # create the queue
        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


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

        self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_fusion)
        # self.fusion_encoder = FusionForMaskedLM(config=self.config_fusion)
        self.text_decoder = BertPrefixModel.from_pretrained(config['text_decoder'], config=self.config_decoder)

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


        self.distill = config.get('distill', True)
        if self.distill:
            self.visual_encoder_m = TimeSformer(
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
            self.text_encoder_m = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder,
                                                            add_pooling_layer=False)
            # self.fusion_encoder_m = FusionForMaskedLM(config=self.config_fusion)
            self.fusion_encoder_m = FusionForMaskedLM.from_pretrained(config['text_decoder'], config=self.config_fusion)
            self.text_decoder_m = BertPrefixModel.from_pretrained(config['text_decoder'], config=self.config_decoder)
            self.vision_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_decoder, self.text_decoder_m],
                                [self.text_proj, self.text_proj_m],
                                [self.vision_proj, self.vision_proj_m],
                                ]
            if self.config_encoder.hidden_size != visual_cfg['embed_dim']:
                self.visn_fc_m = nn.Linear(visual_cfg['embed_dim'], self.config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(self.config_encoder.hidden_dropout_prob)
                self.model_pairs.extend(
                    [[self.visn_fc, self.visn_fc_m], [self.visn_layer_norm, self.visn_layer_norm_m]])
            self.copy_params()
            self.momentum = 0.995


    def forward(self, image, text, alpha=0, prefix_input=None, prefix_target=None):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        _, image_embeds = self.visual_encoder(image)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            _, image_embeds_m = self.visual_encoder_m(image)
            if self.large:
                image_embeds_m = self.dropout_m(self.visn_layer_norm_m(self.visn_fc_m(image_embeds_m)))
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        _, output_pos = self.fusion_encoder.bert(encoder_embeds=text_embeds,
                                                 attention_mask=text.attention_mask,
                                                 encoder_hidden_states=image_embeds,
                                                 encoder_attention_mask=image_atts,
                                                 return_dict=False
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
        _, output_neg = self.fusion_encoder.bert(encoder_embeds=text_embeds_all,
                                                 attention_mask=text_atts_all,
                                                 encoder_hidden_states=image_embeds_all,
                                                 encoder_attention_mask=image_atts_all,
                                                 return_dict=False
                                                 )

        vl_embeddings = torch.cat([output_pos[:, 0, :], output_neg[:, 0, :]], dim=0)
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

        if self.distill:
            with torch.no_grad():
                text_output_m = self.text_encoder_m(input_ids, attention_mask=text.attention_mask,
                                                    return_dict=True)
                text_embeds_m = text_output_m.last_hidden_state
                logits_m = self.fusion_encoder_m(encoder_embeds=text_embeds_m,
                                                 attention_mask=text.attention_mask,
                                                 encoder_hidden_states=image_embeds_m,
                                                 encoder_attention_mask=image_atts,
                                                 return_dict=False,
                                                 return_logits=True
                                                 )
                soft_labels = F.softmax(logits_m, dim=-1)
        else:
            soft_labels = None
        text_output = self.text_encoder(input_ids, attention_mask=text.attention_mask,
                                        return_dict=True)
        text_embeds = text_output.last_hidden_state
        loss_mlm = self.fusion_encoder(encoder_embeds=text_embeds,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=False,
                                       labels=labels,
                                       soft_labels=soft_labels,
                                       alpha=alpha
                                       )
        if self.distill:
            with torch.no_grad():
                text_output_m = self.text_encoder_m(prefix_input.input_ids, attention_mask=prefix_input.attention_mask,
                                                    return_dict=True)
                text_embeds_m = text_output_m.last_hidden_state
                image_embeds_m, text_output_m = self.fusion_encoder_m.bert(encoder_embeds=text_embeds_m,
                                                                           attention_mask=prefix_input.attention_mask,
                                                                           encoder_hidden_states=image_embeds_m,
                                                                           encoder_attention_mask=image_atts,
                                                                           return_dict=False
                                                                           )
                fusion_output_m = torch.cat([image_embeds_m, text_output_m], 1)
                fusion_attention = torch.cat([image_atts, prefix_input.attention_mask], 1)

                logits_m = self.text_decoder_m(prefix_target.input_ids,
                                               attention_mask=prefix_target.attention_mask,
                                               encoder_hidden_states=fusion_output_m,
                                               encoder_attention_mask=fusion_attention,
                                               return_logits=True,
                                               )

        text_output = self.text_encoder(prefix_input.input_ids, attention_mask=prefix_input.attention_mask,
                                        return_dict=True)
        text_output = text_output.last_hidden_state
        image_embeds, text_output = self.fusion_encoder.bert(encoder_embeds=text_output,
                                                             attention_mask=prefix_input.attention_mask,
                                                             encoder_hidden_states=image_embeds,
                                                             encoder_attention_mask=image_atts,
                                                             return_dict=False)
        fusion_output = torch.cat([image_embeds, text_output], 1)
        answer_targets = prefix_target.input_ids.masked_fill(prefix_target.input_ids == self.tokenizer.pad_token_id,
                                                             -100)
        answer_output = self.text_decoder(prefix_target.input_ids,
                                          attention_mask=prefix_target.attention_mask,
                                          encoder_hidden_states=fusion_output,
                                          encoder_attention_mask=fusion_attention,
                                          labels=answer_targets,
                                          return_dict=True,
                                          soft_labels=F.softmax(logits_m, dim=-1) if self.distill else None,
                                          reduction='none',
                                          )
        loss_prefix = answer_output.loss
        return loss_mlm, loss_ita, loss_itm, loss_prefix

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

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



class mPLUG_Cls(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.text_width = self.config_encoder.hidden_size
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])

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

        self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_fusion)

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
        _, image_embeds = self.visual_encoder(image)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_embeds = text_output.last_hidden_state

        ###=================================###
        # forward the positve image-text pair
        _, output_pos = self.fusion_encoder.bert(encoder_embeds=text_embeds,
                                                 attention_mask=text.attention_mask,
                                                 encoder_hidden_states=image_embeds,
                                                 encoder_attention_mask=image_atts,
                                                 return_dict=False
                                                 )

        logits = self.cls_head(output_pos[:, 0, :])
        if labels is None:
            return logits
        else:
            loss = F.cross_entropy(logits, labels)
            return logits, loss

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}



class mPLUG_Caption(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.text_width = self.config_encoder.hidden_size
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decoder_layers

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

        self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_fusion)
        # self.fusion_encoder = FusionForMaskedLM(config=self.config_fusion)
        self.text_decoder = BertPrefixModel.from_pretrained(config['text_decoder'], config=self.config_decoder)

        self.large = False
        if self.config_encoder.hidden_size != visual_cfg['embed_dim']:
            self.visn_fc = nn.Linear(visual_cfg['embed_dim'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        print("large: ", self.large)

        self.beam_generator = TextGenerator(config, self.text_decoder)

    def forward(self, image, text=None, caption=None):
        _, image_embeds = self.visual_encoder(image)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if text is not None:
            text_output = self.text_encoder(text.input_ids, attention_mask=prefix_input.attention_mask,
                                            return_dict=True)
            text_output = text_output.last_hidden_state
            image_embeds, text_output = self.fusion_encoder.bert(encoder_embeds=text_output,
                                                                attention_mask=text.attention_mask,
                                                                encoder_hidden_states=image_embeds,
                                                                encoder_attention_mask=image_atts,
                                                                return_dict=False)
            fusion_output = torch.cat([image_embeds, text_output], 1)
            fusion_attention = torch.cat([image_atts, text.attention_mask], 1)
        else:
            fusion_output = image_embeds
            fusion_attention = image_atts

        if caption is not None:
            answer_targets = caption.input_ids.masked_fill(caption.input_ids == self.tokenizer.pad_token_id,
                                                                -100)
            answer_output = self.text_decoder(caption.input_ids,
                                            attention_mask=caption.attention_mask,
                                            encoder_hidden_states=fusion_output,
                                            encoder_attention_mask=fusion_attention,
                                            labels=answer_targets,
                                            return_dict=True,
                                            reduction='none',
                                            )
            loss = answer_output.loss
            return loss
        else:
            topk_ids, topk_probs = self.generation(fusion_output, fusion_attention) 
            return topk_ids, topk_probs
    
    def generation(self, question_states, question_atts, out_size=1):
        encoder_inputs = [question_states, question_atts]
        topk_ids,topk_probs = self.beam_generator.translate_batch_scst(encoder_inputs,out_size=out_size)  
        return topk_ids, topk_probs

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token', 'visual_encoder.temporal_embed'}


class mPLUG_Retrieval(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        self.embed_dim = config['embed_dim']
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.text_width = self.config_encoder.hidden_size
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])

        # create the queue
        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


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

        self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_fusion)

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

        self.distill = config.get('distill', True)
        if self.distill:
            self.visual_encoder_m = TimeSformer(
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
            self.text_encoder_m = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder,
                                                            add_pooling_layer=False)
            # self.fusion_encoder_m = FusionForMaskedLM(config=self.config_fusion)
            self.fusion_encoder_m = FusionForMaskedLM.from_pretrained(config['text_encoder'], config=self.config_fusion)
            self.vision_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_proj, self.text_proj_m],
                                [self.vision_proj, self.vision_proj_m],
                                ]
            if self.config_encoder.hidden_size != visual_cfg['embed_dim']:
                self.visn_fc_m = nn.Linear(visual_cfg['embed_dim'], self.config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(self.config_encoder.hidden_dropout_prob)
                self.model_pairs.extend(
                    [[self.visn_fc, self.visn_fc_m], [self.visn_layer_norm, self.visn_layer_norm_m]])
            self.copy_params()
            self.momentum = 0.995


    def forward(self, image, text, idx, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        _, image_embeds = self.visual_encoder(image)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            _, image_embeds_m = self.visual_encoder_m(image)
            if self.large:
                image_embeds_m = self.dropout_m(self.visn_layer_norm_m(self.visn_fc_m(image_embeds_m)))
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        ###=================================###
        # forward the positve image-text pair
        _, output_pos = self.fusion_encoder.bert(encoder_embeds=text_embeds,
                                                 attention_mask=text.attention_mask,
                                                 encoder_hidden_states=image_embeds,
                                                 encoder_attention_mask=image_atts,
                                                 return_dict=False
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
        _, output_neg = self.fusion_encoder.bert(encoder_embeds=text_embeds_all,
                                                 attention_mask=text_atts_all,
                                                 encoder_hidden_states=image_embeds_all,
                                                 encoder_attention_mask=image_atts_all,
                                                 return_dict=False
                                                 )

        vl_embeddings = torch.cat([output_pos[:, 0, :], output_neg[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

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