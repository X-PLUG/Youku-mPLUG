# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from typing import Optional, Union, List

import addict
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


##################### Sampling Utils #####################

def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf."""

    filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(filter_, float('-Inf'))


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf."""

    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Filteration based on the cumulative sum.
    filter_ = cumulative_probs > top_p
    # This shift by 1 is weird and I cannot justify it. This existed
    # in the original implementation:
    #   https://github.com/ari-holtzman/degen/blob/master/gen.py
    # and I guess it is needed so keeping it for now.
    filter_[:, 1:] = filter_[:, :-1].clone()
    # Make sure we at least have one token to select from.
    filter_[..., 0] = 0

    # Fill in the filtered part
    filter_ = filter_.scatter(1, sorted_indices, filter_)
    logits.masked_fill_(filter_, float('-Inf'))


def sample(logits, top_k=0, top_p=0.0, temperature=1.0, vocab_size=None):
    """ Sample and generate a token.
    Note: logits has the dimension [b, v] where b is the batch size
          and v is the vocabulary size.
    If vocab_size is provided, we will make sure the sample that is
    generated is in [0, vocab-size). This will avoid out of vocabulary
    generations due to padding.
    """

    # Check logits for consistency.
    assert logits.ndim == 2, 'expected the logits to be of [b, v] shape.'

    # Greedy is just simple argmax.
    if top_k == 1:
        assert top_p == 0.0, 'cannot set both greedy and top-p samplings.'
        samples = torch.argmax(logits, dim=-1)

    # Top-k or top-p sampling.
    else:
        # Clone so we do not modify the inputs,
        logits = logits.clone()
        # Apply temperature in place.
        if temperature != 1.0:
            logits.div_(temperature)

        if top_k > 1:
            assert top_p == 0.0, 'cannot set both top-k and top-p samplings.'
            assert top_k <= logits.size(1), 'top-k is larger than logit size.'
            if vocab_size:
                assert top_k < vocab_size, 'top-k is larger than vocab size.'
            modify_logits_for_top_k_filtering(logits, top_k)

        elif top_p > 0.0:
            assert top_p <= 1.0, 'top-p should be in (0, 1].'
            modify_logits_for_top_p_filtering(logits, top_p)

        # After filtering, we need to recalculate the distribution.
        probs = logits.softmax(dim=-1)
        samples = torch.multinomial(probs, num_samples=1).view(-1)

    # If vocab size is provided, make sure the samples are in
    # in the range [0, vocab-size).
    if vocab_size:
        samples = torch.clamp(samples, min=0, max=(vocab_size - 1))

    return samples



##################### GPT3 (Base, Large) #####################

class GPT3Config(PretrainedConfig):

    model_type = 'gpt3'

    def __init__(
            self,
            vocab_size=25600,
            hidden_size=768,
            ffn_hidden_size=None,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=2048,
            type_vocab_size=2,
            layernorm_epsilon=1e-12,
            bias_gelu_fusion=True,
            fp32_residual_connection=False,
            sequence_parallel=False,
            fp16=False,
            bf16=False,
            apply_query_key_layer_scaling=True,
            attention_softmax_in_fp32=False,
            kv_channels=None,
            masked_softmax_fusion=True,
            attention_dropout=0.1,
            bias_dropout_fusion=True,
            apply_residual_connection_post_layernorm=False,
            hidden_dropout=0.1,
            init_method_std=0.02,
            # generate
            eod_id=7,
            tokens_to_generate=100,
            # top_k=0,
            # top_p=0.9,
            top_k=20,
            top_p=0,
            **kwargs):
        super().__init__(layer_norm_eps=layernorm_epsilon, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = 4 * hidden_size \
            if ffn_hidden_size is None else ffn_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layernorm_epsilon = layernorm_epsilon
        self.bias_gelu_fusion = bias_gelu_fusion
        self.fp32_residual_connection = fp32_residual_connection
        self.sequence_parallel = sequence_parallel
        self.fp16 = fp16
        self.bf16 = bf16
        assert not (fp16 and bf16)
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        if kv_channels is None:
            assert hidden_size % num_attention_heads == 0
            self.kv_channels = hidden_size // num_attention_heads
        self.masked_softmax_fusion = masked_softmax_fusion
        self.attention_dropout = attention_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.apply_residual_connection_post_layernorm = \
            apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.init_method_std = init_method_std
        self.eod_id = eod_id
        self.tokens_to_generate = tokens_to_generate
        self.top_k = top_k
        self.top_p = top_p

        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        self.no_persist_layer_norm = \
            TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11)

    @property
    def params_dtype(self):
        if self.fp16:
            return torch.half
        elif self.bf16:
            return torch.bfloat16
        else:
            return torch.float


class GPT3SelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # Per attention head
        self.hidden_size_per_attention_head = \
            self.hidden_size // self.num_attention_heads

        self.query_key_value = nn.Linear(self.hidden_size,
                                         3 * self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(
            config.attention_probs_dropout_prob)

        # Output.
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (
            self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def _split_tensor_along_last_dim(self,
                                     tensor,
                                     num_partitions,
                                     contiguous_split_chunks=False):
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(self, hidden_states, ltor_mask, is_infer=False):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        tgt_len = hidden_states.size(1)
        ltor_mask = torch.reshape(ltor_mask, [1, 1, tgt_len, tgt_len])
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = \
            self._split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        previous_type = value_layer.type()

        # Raw attention scores. [b, np, s, s]
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)
        # Apply the left to right attention mask.
        if is_infer:
            src_len = key_layer.size(2)
            ltor_mask = torch.tril(
                torch.ones((1, tgt_len, src_len),
                           device=hidden_states.device)).view(
                               1, 1, tgt_len, src_len).type(previous_type)
        converted_mask = 10000.0 * (1.0 - ltor_mask)
        attention_scores = (torch.mul(attention_scores, ltor_mask)
                            - converted_mask).type(previous_type)

        # Attention probabilities. [b, np, s, s]
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size, )
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class GPT3MLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config):
        super().__init__()

        hidden_size = config.hidden_size
        # Project to 4h.
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.activation_func = F.gelu
        # Project back to h.
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GPT3TransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config):
        super().__init__()

        # Layernorm on the input data.
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)

        # Self attention.
        self.attention = GPT3SelfAttention(config)

        # Layernorm on the attention output
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = GPT3MLP(config)

    def forward(self, hidden_states, ltor_mask):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


class GPT3Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super().__init__()

        self.input_tensor = None

        # Number of layers.
        self.num_layers = config.num_hidden_layers

        self.layers = torch.nn.ModuleList(
            [GPT3TransformerLayer(config) for _ in range(self.num_layers)])

        # Final layer norm before output.
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [s, b, h]

        for index in range(self.num_layers):
            layer = self._get_layer(index)
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class GPT3TransformerLanguageModel(nn.Module):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, config):
        super().__init__()

        # Embeddings.
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer.
        self.transformer = GPT3Transformer(config)

    def forward(self, 
        input_ids=None, input_embeds=None, query_embeds=None,
        attention_mask=None, position_ids=None
    ):
        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            words_embeddings = self.word_embeddings(input_ids)
        elif input_embeds is not None:
            words_embeddings = input_embeds
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if query_embeds is not None:
            words_embeddings = torch.cat([query_embeds, words_embeddings], dim=1)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        transformer_input = self.embedding_dropout(embeddings)
        transformer_output = self.transformer(transformer_input,
                                              attention_mask)

        logits = F.linear(transformer_output, self.word_embeddings.weight)
        return logits, transformer_output


class GPT3Model(PreTrainedModel):

    config_class = GPT3Config

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __init__(self, config):
        super().__init__(config)
        self.language_model = GPT3TransformerLanguageModel(config)

    def forward(self,
                input_ids=None,
                input_embeds=None,
                query_embeds=None,
                attention_mask=None,
                position_ids=None,
                labels=None,
                **kwargs):
        if input_ids is not None:
            seq_length = input_ids.size(1)
            device = input_ids.device
        else:
            seq_length = input_embeds.size(1)
            device = input_embeds.device

        if query_embeds is not None:
            seq_length += query_embeds.size(1)

        attention_mask = torch.tril(
            torch.ones((1, 1, seq_length, seq_length),
                       dtype=torch.long,
                       device=device))
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=device)
        
        if query_embeds is None:
            if input_ids is not None:
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            else:
                position_ids = position_ids.unsqueeze(0).expand_as(input_embeds[..., 0])
        else:
            if input_ids is not None:
                position_ids = position_ids.unsqueeze(0).repeat(input_ids.shape[0], 1)
            else:
                position_ids = position_ids.unsqueeze(0).repeat(input_embeds.shape[0], 1)

        logits, last_hidden_state = self.language_model(input_ids, input_embeds, query_embeds, attention_mask, position_ids)
        loss = None
        if labels is not None:
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )
        return addict.Dict(loss=loss, logits=logits, last_hidden_state=last_hidden_state)

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Optional[Union[str,
                                                               os.PathLike]]):
        config = cls.config_class.from_pretrained(
            pretrained_model_name_or_path)
        model = cls(config)
        state_dict_file = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        state_dict = torch.load(state_dict_file)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {
            k.replace('model.language_model', 'language_model'): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        return model

    def generate(self, tokens, query_embeds=None, temperature=1.0, **kwargs):

        batch_size = tokens.size(0)
        lengths = kwargs.pop(
            'prompt_length',
            torch.tensor([tokens.size(1)], device=tokens.device))

        min_prompt_length = lengths.min().item()
        max_sequence_length = tokens.size(1)
        max_sequence_length = min(max_sequence_length,
                                  self.config.max_position_embeddings)

        # If the context is too big, this happens
        if min_prompt_length >= max_sequence_length:
            raise ValueError('context length + tokens_to_generate too large')

        # Added termination_id to support the case that we want to terminate the
        # generation once that id is generated.
        termination_id = self.config.eod_id

        # Whether we have reached a termination id.
        is_generation_done = torch.zeros(
            batch_size, dtype=torch.uint8, device=tokens.device)

        with torch.no_grad():
            for context_length in range(min_prompt_length,
                                        max_sequence_length):

                # Pick the slice that we need to pass through the network.
                tokens2use = tokens[:, :context_length]

                # # logits will be meanigful only in the last pipeline stage.
                # logits = self(tokens2use).logits

                # token2use_embeds = self.language_model.word_embeddings(tokens2use)
                # if query_embeds is not None:
                #     input_embeds = torch.cat([query_embeds, token2use_embeds], dim=1)
                # else:
                #     input_embeds = token2use_embeds
                # attention_mask = torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(tokens.device)
                logits = self(input_ids=tokens2use, query_embeds=query_embeds).logits

                # Sample.
                last_token_logits = logits[:, -1, :]
                new_sample = sample(
                    last_token_logits,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    temperature=temperature,
                    vocab_size=self.config.vocab_size)

                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]

                done_token = (new_sample == termination_id).byte() & \
                    started.byte()

                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)

                if done:
                    break

        tokens = tokens[:, :(context_length + 1)]
        return tokens
