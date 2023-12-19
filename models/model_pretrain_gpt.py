'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vision_transformer import (
    VisionTransformerForMaskedImageModeling, LayerNormWithForceFP32, 
    _convert_pretrained_vit, AttentionPool
)
from models.modelling_gpt2 import GPT2Config, GPT2LMHeadModel, GPT2LMHeadMultiModalModel
from models.visual_transformers import initialize_clip
from models.distributed_utils import all_gather

import torch
import torch.nn.functional as F
from torch import nn

import timm
from timm.models.layers import trunc_normal_

import numpy as np
import random
import math


class MPLUG_COCA(nn.Module):
    def __init__(self, config = None, tokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer

        visual_cfg = config['visual_cfg']
        self.config_decoder = GPT2Config.from_json_file(config['gpt2_config'])
        self.config_decoder.num_hidden_layers = self.config_decoder.n_layer

        self.config_multimodal_decoder = GPT2Config.from_json_file(config['gpt2_config'])
        self.config_multimodal_decoder.num_hidden_layers = self.config_multimodal_decoder.n_layer
        self.max_position_embeddings = self.config_multimodal_decoder.n_positions
        self._triangle_mask = torch.tril(torch.ones((self.max_position_embeddings, self.max_position_embeddings), dtype=torch.long, requires_grad=False))

        self.text_width = self.config_multimodal_decoder.hidden_size
        self.vision_width = visual_cfg['embed_dim']

        self.only_masked = config.get('only_masked', False)

        self.visual_encoder = VisionTransformerForMaskedImageModeling(
            img_size=visual_cfg['img_size'], patch_size=visual_cfg['patch_size'], 
            embed_dim=visual_cfg['embed_dim'], depth=visual_cfg['depth'], 
            num_heads=visual_cfg['num_heads'],
            mlp_ratio=visual_cfg['mlp_ratio'], qkv_bias=True,
            use_mean_pooling=visual_cfg.get('use_mean_pooling', False),
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

        self.visual_lm_head = nn.Linear(self.text_width, visual_cfg['predict_feature_dim'])
        trunc_normal_(self.visual_lm_head.weight, std=0.015)

        if self.text_width != self.vision_width:
            self.visual_fc = nn.Linear(self.vision_width, self.text_width)
            self.visual_norm = partial(LayerNormWithForceFP32, eps=1e-6)(self.text_width)
            trunc_normal_(self.visual_fc.weight, std=0.015)

        self.text_decoder = GPT2LMHeadModel.from_pretrained(config['text_decoder'], config=self.config_decoder)
        self.multimodal_decoder = GPT2LMHeadMultiModalModel(config=self.config_multimodal_decoder)
        # self.token_type_embeddings = nn.Embedding(config.get("token_types", 3), self.text_width)
        # trunc_normal_(self.token_type_embeddings.weight, std=0.015)

    def forward(self, image, text, bool_masked_pos=None, image_target=None):
        
        _, image_embeds = self.visual_encoder(image, bool_masked_pos=None)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) 
        _, masked_image_embeds = self.visual_encoder(image, bool_masked_pos)
        
        # Text Decoder Forward
        text_embeds, _ = self.text_decoder(text.input_ids, attention_mask=text.attention_mask)

        ## Multi-Modal Decoder Forward
        if self.text_width != self.vision_width:
            image_embeds = self.visual_fc(image_embeds)
            image_embeds = self.visual_norm(image_embeds)

            masked_image_embeds = self.visual_fc(masked_image_embeds)
            masked_image_embeds = self.visual_norm(masked_image_embeds)
        
        # Should we add token types here? 
        # visual_token_type = torch.zeros(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        # image_embeds = image_embeds + self.token_type_embeddings(visual_token_type)
        # text_token_type = torch.ones(text_embeds.size()[:-1], dtype=torch.long, device=text_embeds.device)
        # text_embeds = text_embeds + self.token_type_embeddings(text_token_type)

        # Image Caption:
        input_embedding = torch.cat([image_embeds, text_embeds], dim=1)
        causal_mask_caps = self.create_causal_mask(image_embeds.shape[1], text_embeds.shape[1], mask_t2v=False, mask_v2t=True).to(image.device)
        attn_mask = torch.cat([image_atts, text.attention_mask], dim=1)
        _, prediction_scores = self.multimodal_decoder(
            inputs_embeds=input_embedding,
            # encoder_embeds=input_embedding,
            attention_mask=attn_mask,
            causal_mask=causal_mask_caps,
            mode="text",
        )

        prediction_scores = prediction_scores[:, image_embeds.shape[1]:] # Seperate text score
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous() 
        labels = text.input_ids[:, 1:].contiguous()
        labels = labels.masked_fill(~text.attention_mask[:, 1:].bool(), -100)  # Ignore Unused

        loss_caption = F.cross_entropy(shifted_prediction_scores.view(-1, self.config_multimodal_decoder.vocab_size), labels.view(-1))

        # Mask Visual Modeling
        input_embedding = torch.cat([masked_image_embeds, text_embeds], dim=1)
        # causal_mask_mim = self.create_causal_mask(image_embeds.shape[1], text_embeds.shape[1], mask_t2v=False, mask_v2t=False).to(image.device)
        causal_mask_mim = torch.ones((input_embedding.shape[1], input_embedding.shape[1]), dtype=torch.long).to(image.device) 
        output_features, _ = self.multimodal_decoder(
            inputs_embeds=input_embedding,
            # encoder_embeds=input_embedding,
            attention_mask=attn_mask,
            causal_mask=causal_mask_mim,
            mode="vision",
        )

        output_features = output_features[:, 1:image_embeds.shape[1]] # Seperate image feature and remove CLS
        predicted_features = self.visual_lm_head(output_features[bool_masked_pos])
        image_target = image_target[bool_masked_pos]

        loss_mim = 1 - F.cosine_similarity(predicted_features.float(), image_target.float(), dim=-1).mean()

        return loss_caption, loss_mim


    @torch.no_grad()
    def create_causal_mask(self, visual_seq_len, text_seq_len, mask_t2v=False, mask_v2t=True):
        curr_len = visual_seq_len + text_seq_len
       
        attention_mask = torch.zeros((curr_len, curr_len), dtype=torch.long)

        # Q_v K_v & Q_t, K_t
        attention_mask[:visual_seq_len, :visual_seq_len] = 1
        attention_mask[visual_seq_len:, visual_seq_len:].copy_(
            self._triangle_mask[:text_seq_len, :text_seq_len]
        )
        # Q_t K_v = 1; Q_v K_t = 0
        if not mask_t2v:
            attention_mask[visual_seq_len:, :visual_seq_len] = 1
        if not mask_v2t:
            attention_mask[:visual_seq_len, visual_seq_len:] = 1
        return attention_mask


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'visual_encoder.pos_embed', 'visual_encoder.cls_token'}


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