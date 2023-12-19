# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors. All rights reserved.
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

import io
import math
import os
from os import path as osp
from typing import Callable, Dict, List, Optional, Union, Any

import torch
import addict
from megatron_util import mpu
from megatron_util.global_vars import get_global_memory_buffer
from megatron_util.model import (AttnMaskType, Float16Module, LayerNorm,
                                 bias_gelu_impl)
from megatron_util.model.fused_softmax import FusedScaleMaskSoftmax
from torch import nn
import numpy as np
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from tokenizers import Tokenizer

from utils import File
from megatron_util import initialize_megatron


########################## Tokenizer #################################

class JiebaBPETokenizer:
    """SentencePiece BPE tokenizer with Jieba integration"""

    def __init__(self, tokenizer_json_file):
        self.name = 'Jieba BPE Tokenizer'

        self.tokenizer = Tokenizer.from_file(tokenizer_json_file)
        self.eod_id = self.tokenizer.token_to_id('<|endoftext|>')
        try:
            import jieba
            import logging
            jieba.setLogLevel(logging.INFO)
        except ImportError:
            raise ImportError(
                'You need to install jieba to use JiebaTokenizer. '
                'See https://pypi.org/project/rjieba/ for installation.')
        self.jieba = jieba
        self.new_line = self.vocab['\n']
        self.sep_token = self.vocab['<sep>']
        # From Wangwei: <sep> 分割doc，</s> 专用于对话分割一个来回的对话
        self.bos_id = self.tokenizer.token_to_id('<sep>')
        self.pad_id = self.tokenizer.token_to_id('<|endoftext|>')
        self.eos_id = self.tokenizer.token_to_id('<|endoftext|>')

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def vocab(self):
        return self.tokenizer.get_vocab(with_added_tokens=True)

    @property
    def inv_vocab(self):
        vocab = self.vocab
        inv_vocab = dict()
        for key, val in vocab.items():
            inv_vocab[val] = key
        return inv_vocab

    def tokenize(self, text: str, is_code: bool = False, add_special_tokens=True) -> List[int]:
        """
        """
        if not is_code:
            seg_list = [x for x in self.jieba.cut(text)]
            token_ids = self.tokenizer.encode(
                seg_list, is_pretokenized=True, add_special_tokens=True).ids 
        else:
            token_ids = self.tokenizer.encode(
                text, is_pretokenized=False, add_special_tokens=True).ids
        
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        return token_ids

    def tokenize_prompt(self, prompt_text: str, text: str, is_code: bool = False, add_special_tokens=True) -> List[int]:
        """
        """
        if not is_code:
            seg_list = [x for x in self.jieba.cut(text)]
            seg_list_prompt = [x for x in self.jieba.cut(prompt_text)]
            token_ids = self.tokenizer.encode(
                seg_list, is_pretokenized=True, add_special_tokens=True).ids 
            token_ids_prompt = self.tokenizer.encode(
                seg_list_prompt, is_pretokenized=True, add_special_tokens=True).ids 
        else:
            token_ids = self.tokenizer.encode(
                text, is_pretokenized=False, add_special_tokens=True).ids
            token_ids_prompt = self.tokenizer.encode(
                prompt_text, is_pretokenized=False, add_special_tokens=True).ids
        
        if add_special_tokens:
            token_ids = [[self.bos_id], token_ids_prompt, token_ids, [self.eos_id]]
        return token_ids

    def detokenize(self, token_ids):
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    @property
    def eod(self):
        return self.eod_id
    
    @property
    def eos(self):
        return self.eos_id
    
    @property
    def bos(self):
        return self.bos_id

    @property
    def pad(self):
        return self.pad_id


class BatchEncoding:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item: Union[int, str]) -> Union[Any]:
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError(
                "Indexing with integers (to access backend Encoding for a given batch index) "
                "is not available when using Python based tokenizers"
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data}

    def __repr__(self):
        return str(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()
    
    def to(self, device: Union[str, "torch.device"]):
        if isinstance(device, str) or isinstance(device, torch.device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self

class DistributedGPT3Tokenizer:
    def __init__(self, model_dir: str, sequence_length: int = 128):
        self.tokenizer = JiebaBPETokenizer(osp.join(model_dir, 'tokenizer.json'))
        self.max_length = sequence_length

    def decode(self, tokens, **kwargs):
        """Decode the tokens to real text.

        Args:
            tokens: The output tokens from model's `forward` and `generate`

        Returns:
            The actual text.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return self.tokenizer.detokenize(tokens)

    def _truncate(self, array: np.ndarray, max_length: int=None) -> np.ndarray:
        if max_length is None:
            max_length = self.max_length

        if len(array) < max_length:
            return np.pad(
                array, (0, max_length - len(array)),
                constant_values=self.tokenizer.pad), len(array)
        else:
            return array[:max_length], max_length
    
    def _truncate_prompt(self, array, max_length: int=None) -> np.ndarray:
        if max_length is None:
            max_length = self.max_length
        
        total_length = sum([len(a) for a in array])
        prompt_length = len(array[1])

        if total_length < max_length:
            bos, prompt_text, text, eos = array
            res = np.array(bos + prompt_text + text + eos)
            return np.pad(
                res, (0, max_length - total_length),
                constant_values=self.tokenizer.pad), prompt_length, total_length
        else:
            bos, prompt_text, text, eos = array
            if len(prompt_text) >= max_length-len(text)-2 and max_length-len(text)-2 >= 0:
                prompt_text = prompt_text[:max_length-len(text)-2]
                prompt_length = len(prompt_text)
            else:
                print("Truncate Target")
                text = text[:max_length-2-len(prompt_text)]
                prompt_length = len(prompt_text)
                # raise NotImplementedError
            res = np.array(bos + prompt_text + text + eos)
            return res, prompt_length, max_length

    def __call__(self, data, padding='longest', truncation=True, max_length=None, return_tensors='pt', add_special_tokens=True, **kwargs):
        if isinstance(data[0], str):
            max_num_tokens = 0
            tokenized_data = []
            input_ids = []
            attention_mask = []
            for text in data:
                tokens = self.tokenizer.tokenize(text, add_special_tokens)
                max_num_tokens = max(max_num_tokens, len(tokens))
                tokenized_data.append(tokens)
            
            for d in tokenized_data:
                if truncation:
                    if padding == 'longest':
                        out, out_len = self._truncate(np.array(d), min(max_num_tokens, max_length))
                        mask = np.zeros(min(max_num_tokens, max_length))
                        mask[:out_len] = 1
                    elif padding == 'max_length':
                        out, out_len = self._truncate(np.array(d), max_length)
                        mask = np.zeros(max_length)
                        mask[:out_len] = 1
                else:
                    if padding == 'longest':
                        out, out_len = self._truncate(np.array(d), max_num_tokens)
                        mask = np.zeros(max_num_tokens)
                        mask[:out_len] = 1
                    elif padding == 'max_length':
                        out, out_len = self._truncate(np.array(d), max_length)
                        mask = np.zeros(max_length)
                        mask[:out_len] = 1
                input_ids.append(out)                  
                attention_mask.append(mask.astype(int))
            
            input_ids = np.stack(input_ids, axis=0)
            attention_mask = np.stack(attention_mask, axis=0)

            if return_tensors == 'pt':
                input_ids = torch.tensor(input_ids).long()
                attention_mask = torch.tensor(attention_mask).long()

            output = BatchEncoding(dict(input_ids=input_ids, attention_mask=attention_mask))
        else:
            max_num_tokens = 0
            tokenized_data = []
            input_ids = []
            attention_mask = []
            prompt_lengths = []

            for prompt_text, text in data:
                tokens = self.tokenizer.tokenize_prompt(prompt_text, text, add_special_tokens)
                max_num_tokens = max(max_num_tokens, sum([len(x) for x in tokens]))
                tokenized_data.append(tokens)
            
            for d in tokenized_data:
                if truncation:
                    if padding == 'longest':
                        out, prompt_length, out_len = self._truncate_prompt(d, max_length)
                        mask = np.zeros(max_length)
                        mask[:out_len] = 1
                    elif padding == 'max_length':
                        out, prompt_length, out_len = self._truncate_prompt(d, max_length)
                        mask = np.zeros(max_length)
                        mask[:out_len] = 1
                else:
                    if padding == 'longest':
                        out, prompt_length, out_len = self._truncate_prompt(d, max_num_tokens)
                        mask = np.zeros(max_num_tokens)
                        mask[:out_len] = 1
                    elif padding == 'max_length':
                        out, prompt_length, out_len = self._truncate_prompt(d, max_length)
                        mask = np.zeros(max_length)
                        mask[:out_len] = 1

                input_ids.append(out) 
                attention_mask.append(mask.astype(int))
                prompt_lengths.append(prompt_length)
            
            if return_tensors == 'pt':
                input_ids = torch.tensor(input_ids).long()
                attention_mask = torch.tensor(attention_mask).long()
                prompt_lengths = torch.tensor(prompt_lengths).long()

            output = BatchEncoding(dict(input_ids=input_ids, attention_mask=attention_mask, prompt_lengths=prompt_lengths))

        return output


##################### TorchModel ######################
class TorchModel(torch.nn.Module):
    """ Base model interface for pytorch
    """

    def __init__(self, model_dir=None, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        torch.nn.Module.__init__(self)

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        # Adapting a model with only one dict arg, and the arg name must be input or inputs
        if func_receive_dict_inputs(self.forward):
            return self.postprocess(self.forward(args[0], **kwargs))
        else:
            return self.postprocess(self.forward(*args, **kwargs))

    def _load_pretrained(self,
                         net,
                         load_path,
                         strict=True,
                         param_key='params'):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info(
                    f'Loading: {param_key} does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].'
        )
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)
        logger.info('load model done.')
        return net

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def post_init(self):
        """
        A method executed at the end of each model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



##################### Utils ############################

# from modelscope.utils.checkpoint import weights_to_cpu
# from modelscope.utils.megatron_utils import init_megatron_util
# from modelscope.utils.nlp.load_checkpoint import pre_load

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


# Overwrite the function from modelscope
def init_megatron_util(megatron_cfg: Optional[Dict] = None, **kwargs):
    if megatron_cfg is None:
        megatron_cfg = {
            "world_size": 1,
            "model_parallel_size": 8,
            "tensor_model_parallel_size": 8
        }
    megatron_cfg.update(kwargs)
    initialize_megatron(megatron_cfg)


def _get_ckpt_name(mp_rank, checkpoints_path, tag):
    ckpt_name = os.path.join(
        checkpoints_path, str(tag),
        'mp_rank_{:02d}'.format(mp_rank) + '_model_states.pt')
    return ckpt_name

def pre_load(mp_rank, load_dir, tag=''):
    load_path = _get_ckpt_name(mp_rank, load_dir, tag)
    checkpoint = torch.load(
        load_path, map_location=lambda storage, loc: storage)
    return checkpoint['module']


def _get_ckpt_name_larger(mp_rank, checkpoints_path, tag):
    ckpt_name = os.path.join(
        checkpoints_path, str(tag),
        'mp_rank_{:02d}'.format(mp_rank), 'model_optim_rng.pt')
    return ckpt_name

def pre_load_larger(mp_rank, load_dir, tag=''):
    load_path = _get_ckpt_name_larger(mp_rank, load_dir, tag)
    checkpoint = torch.load(
        load_path, map_location=lambda storage, loc: storage)
    return checkpoint['model']


##################### GPT3 (1.3B +) #####################

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
            top_k=0,
            top_p=0.9,
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


class GPT3ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, init_method, output_layer_init_method):
        super().__init__()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = config.bias_gelu_fusion
        self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(
            hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = \
                bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class GPT3Embedding(nn.Module):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, config, init_method):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.init_method = init_method

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            config.vocab_size, self.hidden_size, init_method=self.init_method)

        # Position embedding (serial).
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                self.hidden_size)
        # Initialize the position embeddings.
        self.init_method(self.position_embeddings.weight)

        self.fp32_residual_connection = config.fp32_residual_connection
        self.sequence_parallel = config.sequence_parallel
        # Embeddings dropout
        self.embedding_dropout = nn.Dropout(config.hidden_dropout)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True

    def forward(self, input_ids, input_embeds, query_embeds, position_ids):
        # Embeddings.
        if input_ids is not None:
            words_embeddings = self.word_embeddings(input_ids)
        else:
            words_embeddings = input_embeds
        if query_embeds is not None:
            words_embeddings = torch.cat([query_embeds, words_embeddings], dim=1)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = mpu.scatter_to_sequence_parallel_region(embeddings)
            with mpu.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)
        return embeddings


class NoopTransformerLayer(nn.Module):

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self,
                hidden_states,
                attention_mask,
                encoder_output=None,
                enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


class GPT3CoreAttention(nn.Module):

    def __init__(self,
                 config,
                 layer_number,
                 attn_mask_type=AttnMaskType.padding):
        super().__init__()
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            config.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16, self.attn_mask_type,
            config.masked_softmax_fusion, attention_mask_func,
            self.attention_softmax_in_fp32, coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2),
                       query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, 'mpu')

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2),
                       query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class GPT3ParallelAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, init_method, output_layer_init_method,
                 layer_number):
        super().__init__()
        self.layer_number = max(1, layer_number)
        self.params_dtype = config.params_dtype

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            config.num_attention_heads, world_size)

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            config.hidden_size,
            3 * projection_size,
            gather_output=False,
            init_method=init_method)

        self.core_attention = GPT3CoreAttention(config, self.layer_number)

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask, inference_params=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
             3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer,
         value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end,
                                             batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end,
                                                 batch_start:batch_end, ...]

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer,
                                            value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


class nullcontext:

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = F.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):

    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor, bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor, bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class GPT3ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, init_method, output_layer_init_method,
                 layer_number):

        super().__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=config.no_persist_layer_norm,
            sequence_parallel=config.sequence_parallel)

        # Self attention.
        self.self_attention = GPT3ParallelAttention(config, init_method,
                                                    output_layer_init_method,
                                                    layer_number)
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=config.no_persist_layer_norm,
            sequence_parallel=config.sequence_parallel)

        # MLP
        self.mlp = GPT3ParallelMLP(config, init_method,
                                   output_layer_init_method)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1
                                          and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
            nullcontext if use_nvfuser else torch.enable_grad

    def forward(self, hidden_states, attention_mask, inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(
                attention_output, attention_bias.expand_as(residual), residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        with self.bias_dropout_add_exec_handler():
            output = bias_dropout_add_func(mlp_output,
                                           mlp_bias.expand_as(residual),
                                           residual, self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = mpu.make_viewless_tensor(
            inp=output, requires_grad=output.requires_grad, keep_graph=True)

        return output


class GPT3ParallelTransformer(nn.Module):
    """Transformer class."""

    def __init__(self,
                 config,
                 init_method,
                 output_layer_init_method,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True):
        super().__init__()

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None

        self.sequence_parallel = config.sequence_parallel

        # Number of layers.
        self.num_layers = config.num_hidden_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GPT3ParallelTransformerLayer(config, init_method,
                                                output_layer_init_method,
                                                layer_number)

        if self.num_layers == 0:
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                no_persist_layer_norm=config.no_persist_layer_norm,
                sequence_parallel=config.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask, inference_params=None):
        # hidden_states: [s, b, h]

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = mpu.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.sequence_parallel:
            rng_context = mpu.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # Forward pass.
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    inference_params=inference_params)

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
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

    def __init__(self, config, init_method, output_layer_init_method):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.init_method = init_method
        self.encoder_hidden_state = None

        # Embeddings.
        self.embedding = GPT3Embedding(config, self.init_method)

        # Transformer.
        self.encoder = GPT3ParallelTransformer(
            config,
            self.init_method,
            output_layer_init_method,
        )

    def forward(self,
                enc_input_ids,
                enc_input_embeds,
                query_embeds,
                enc_position_ids,
                enc_attn_mask,
                inference_params=None,
                enc_hidden_states=None):

        # Encoder embedding.
        if enc_input_ids is not None:
            encoder_input = self.embedding(enc_input_ids, None, query_embeds, enc_position_ids)
        elif enc_input_embeds is not None:
            encoder_input = self.embedding(None, enc_input_embeds, query_embeds, enc_position_ids)
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    inference_params=inference_params)
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        return encoder_output


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT3Model(PreTrainedModel):

    config_class = GPT3Config

    def __init__(self, config):
        super().__init__(config)

        self.language_model = GPT3TransformerLanguageModel(
            config, init_method_normal(config.init_method_std),
            scaled_init_method_normal(config.init_method_std,
                                      config.num_hidden_layers))

    def word_embeddings_weight(self):
        return self.language_model.embedding.word_embeddings.weight

    @staticmethod
    def build_attention_mask_and_position_ids(tokens):
        seq_length = tokens.size(1)
        attention_mask = torch.tril(
            torch.ones((1, 1, seq_length, seq_length), device=tokens.device))
        attention_mask = (attention_mask < 0.5)

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        return attention_mask, position_ids

    @staticmethod
    def build_position_ids(tokens):
        seq_length = tokens.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        return attention_mask, position_ids

    def forward(self,
                input_ids=None,
                input_embeds=None,
                query_embeds=None,
                attention_mask=None,
                position_ids=None,
                inference_params=None,
                labels=None,
                **kwargs):

        if input_ids is not None and input_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            if attention_mask is None and position_ids is None:
                attention_mask, position_ids = \
                    self.build_attention_mask_and_position_ids(input_ids)
            elif position_ids is None:
                position_ids = self.build_position_ids(input_ids)
        elif input_embeds is not None:
            if attention_mask is None and position_ids is None:
                attention_mask, position_ids = \
                    self.build_attention_mask_and_position_ids(input_embeds[..., 0])
            elif position_ids is None:
                position_ids = self.build_position_ids(input_embeds[..., 0])
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        lm_output = self.language_model(
            input_ids,
            input_embeds,
            query_embeds,
            position_ids,
            attention_mask,
            inference_params=inference_params)

        logits_parallel = mpu.LinearWithGradAccumulationAndAsyncCommunication.apply(
            lm_output, self.word_embeddings_weight(), None, False, True,
            self.config.sequence_parallel)

        losses = None
        if labels is not None:
            # [b s] => [s b]
            labels = labels.transpose(0, 1).contiguous()
            losses = mpu.vocab_parallel_cross_entropy(
                logits_parallel.clone().float(), labels)
            # [s b] => [b s]
            losses = losses.transpose(0, 1).contiguous()

        # Gather if needed.
        logits = mpu.gather_from_tensor_model_parallel_region(logits_parallel)
        # [s b h] => [b s h]
        logits = logits.transpose(0, 1).contiguous()

        return logits, losses, lm_output.transpose(0, 1).contiguous()


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


class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_len):
        """Note that offsets are set to zero and we always set the
        flag to allocate memory. After the first call, make sure to
        set this flag to False."""
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_idx):
        'swap between batches'
        if len(self.key_value_memory_dict) == 0:
            raise ValueError('should not swap when dict in empty')

        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[
                layer_number]
            assert len(batch_idx) == inference_key_memory.shape[
                1]  # make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                new_inference_key_memory, new_inference_value_memory)


def split_into_partitions(tensor, num_partitions, partition_dim, stride):
    per_partition_size = mpu.utils.divide(
        tensor.size(partition_dim), num_partitions)
    per_partition_per_stride_size = mpu.utils.divide(per_partition_size,
                                                     stride)
    partitions_list = torch.split(
        tensor, per_partition_per_stride_size, dim=partition_dim)
    partitions = []
    for i in range(num_partitions):
        partition = torch.cat(
            partitions_list[i::num_partitions], dim=partition_dim)
        partitions.append(partition)
    return partitions


def split_state_dict(state_dict: Dict[str, torch.Tensor], model: GPT3Model,
                     partitions: int) -> Dict[str, torch.Tensor]:
    if partitions == 1:
        return state_dict
    rank: int = mpu.get_tensor_model_parallel_rank()
    for name, parameters in model.named_parameters():
        if parameters.shape == state_dict[name].shape:
            continue
        dim = max(parameters.partition_dim, 0)
        stride = parameters.partition_stride
        state_dict[name] = split_into_partitions(state_dict[name], partitions,
                                                 dim, stride)[rank]
    return state_dict


def save_checkpoint(model: torch.nn.Module, filename: str, **kwargs) -> None:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    checkpoint = {'module': weights_to_cpu(model.state_dict())}
    mp_rank = mpu.get_tensor_model_parallel_rank()
    filename = osp.join(
        osp.dirname(filename), 'model',
        'mp_rank_{:02d}'.format(mp_rank) + '_model_states.pt')

    with io.BytesIO() as f:
        torch.save(checkpoint, f)
        File.write(f.getvalue(), filename)


# class DistributedGPT3(TorchModel):
class DistributedGPT3(nn.Module):

    def __init__(self,
                 model_dir,
                 rank,
                 path_load_tag='model',
                 *args,
                 **kwargs):
        # super().__init__(model_dir, *args, **kwargs)
        super().__init__()

        megatron_cfg = kwargs.pop('megatron_cfg', None)
        init_megatron_util(megatron_cfg=megatron_cfg, rank=rank)

        self.config = GPT3Config.from_pretrained(model_dir)
        # Build model.
        model = GPT3Model(self.config)

        for param in model.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if self.config.fp16 or self.config.bf16:
            model = Float16Module(model, self.config)

        self.dist_model = model

        # For 1.3B and 2.7B, we have to set checkpoint_model_parallel_size=1
        tensor_ws = mpu.get_tensor_model_parallel_world_size() # 8 
        ckpt_ws = kwargs.pop('checkpoint_model_parallel_size', tensor_ws) # 8
        ckpt_rank = mpu.get_tensor_model_parallel_rank() * ckpt_ws // tensor_ws

        load_state_dict = kwargs.pop('load_state_dict', True)
        if load_state_dict:
            if ckpt_ws == 1:
                load_model = pre_load(ckpt_rank, model_dir, tag=path_load_tag)
                load_model = split_state_dict(load_model, model, tensor_ws // ckpt_ws)
            
                self.dist_model.load_state_dict(load_model)
            else:
                pass
            # load_model = pre_load_larger(ckpt_rank, model_dir, tag=path_load_tag)
            # load_model = split_state_dict(load_model, model, tensor_ws // ckpt_ws)
            
            # self.dist_model.load_state_dict(load_model)
        
        self.inference_params = None

    def train(self, mode: bool = True):
        if mode:
            self.inference_params = None
        return super().train(mode)

    def forward(self,
                tokens=None,
                input_embeds=None,
                query_embeds=None,
                attention_mask=None,
                position_ids=None,
                labels=None,
                prompt_length=None,
                loss_mask=None,
                is_pair=(False, )):

        logits, losses, lm_output = self.dist_model(
            tokens,
            input_embeds,
            query_embeds,
            attention_mask,
            position_ids,
            inference_params=self.inference_params,
            labels=labels
        )

        loss = None
        if labels is None:
            self.inference_params.sequence_len_offset += tokens.size(1)
            if query_embeds is not None:
                self.inference_params.sequence_len_offset += query_embeds.size(1)
        else:
            # loss_mask = torch.ones(
            #     tokens.size(), dtype=torch.float, device=tokens.device)

            # if is_pair[0]:
            #     for i, length in enumerate(prompt_length):
            #         loss_mask[i, :length] = 0

            if loss_mask is None:
                loss_mask = attention_mask[:, 1:].contiguous()

            losses = losses[:, :-1].contiguous().float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return addict.Dict(logits=logits, loss=loss, losses=losses, last_hidden_state=lm_output)

    def sample(self,
               tokens,
               query_embeds=None,
               temperature=1.0,
               use_eod_token_for_early_termination=True,
               stop_on_double_eol=False,
               stop_on_eol=False,
               termination_id=None,
               **kwargs):
        batch_size = tokens.size(0)
        lengths = kwargs.pop(
            'prompt_length',
            torch.tensor([tokens.size(1)], device=tokens.device))
        pads = torch.ones(
            batch_size, self.config.tokens_to_generate,
            device=tokens.device).long() * self.config.eod_id
        tokens = torch.cat((tokens, pads), dim=-1)

        min_prompt_length = lengths.min().item()
        max_sequence_length = tokens.size(1)
        max_sequence_length = min(max_sequence_length,
                                  self.config.max_position_embeddings)

        # If the context is too big, this happens
        if min_prompt_length >= max_sequence_length:
            raise ValueError('context length + tokens_to_generate too large')

        # Initialize inference parameters.
        if query_embeds is None:
            self.inference_params = InferenceParams(batch_size,
                                                    max_sequence_length)
        else:
            self.inference_params = InferenceParams(batch_size,
                                                    max_sequence_length + query_embeds.size(1))

        # Added termination_id to support the case that we want to terminate the
        # generation once that id is generated.
        if termination_id is None:
            termination_id = self.config.eod_id

        # Whether we have reached a termination id.
        is_generation_done = torch.zeros(
            batch_size, dtype=torch.uint8, device=torch.cuda.current_device())

        # =============
        # Run infernece
        # =============
        if query_embeds is None:
            attention_mask, position_ids = \
                GPT3Model.build_attention_mask_and_position_ids(tokens)
        else:
            attention_mask, position_ids = \
                GPT3Model.build_attention_mask_and_position_ids(torch.cat([query_embeds[...,0], tokens.float()], dim=1))

        prev_context_length = 0
        total_min_prompt_length = min_prompt_length if query_embeds is None else min_prompt_length + query_embeds.size(1)
        total_max_sequence_length = max_sequence_length if query_embeds is None else max_sequence_length + query_embeds.size(1)
        for context_length in range(total_min_prompt_length, total_max_sequence_length):

            # Pick the slice that we need to pass through the network.
            if query_embeds is None:
                tokens2use = tokens[:, prev_context_length:context_length]
            else:
                tokens2use = tokens[:, max(prev_context_length-query_embeds.size(1), 0):context_length-query_embeds.size(1)]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # logits will be meanigful only in the last pipeline stage.
            logits = self(
                tokens=tokens2use, 
                query_embeds=query_embeds if context_length == total_min_prompt_length else None, 
                attention_mask=attention_mask2use, 
                position_ids=positions2use
            ).logits

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
            if query_embeds is None:
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]
            else:
                started = lengths <= context_length - query_embeds.size(1)
                # Update the tokens.
                tokens[started, context_length-query_embeds.size(1)] = new_sample[started]

            # Update the context length for the next token generation.
            prev_context_length = context_length

            # instead tokenization should be in the inference loop so stop sequences can be used
            if stop_on_double_eol:
                hit_double_eol = (new_sample == 628).byte() & started.byte()
                hit_two_eols = (new_sample == 198).byte() & (
                    tokens[:,
                           context_length - 1] == 198).byte() & started.byte()
                done_token = hit_double_eol | hit_two_eols
            elif stop_on_eol:
                hit_double_eol = (new_sample == 628).byte() & started.byte()
                hit_eol = (new_sample == 198).byte() & started.byte()
                done_token = hit_double_eol | hit_eol
            else:
                done_token = (new_sample == termination_id).byte() & \
                    started.byte()

            is_generation_done = is_generation_done | done_token
            done = torch.all(is_generation_done)

            if use_eod_token_for_early_termination and done:
                break

        tokens = tokens[:, :(context_length + 1)]
        return tokens

    def beam_search(self, tokens, query_embeds=None, beam_size=5, num_return_gen=1, stop_token=None, **kwargs):
        batch_size = tokens.size(0)
        assert (batch_size == 1)
        prompt_length = kwargs.pop(
            'prompt_length',
            torch.tensor([tokens.size(1)], device=tokens.device)).item()
        if stop_token is None:
            stop_token = self.config.eod_id
        pads = torch.ones(
            1, self.config.tokens_to_generate,
            device=tokens.device).long() * stop_token
        tokens = torch.cat((tokens, pads), dim=-1)
        final_sequence_length = tokens.size(1)
        final_sequence_length = min(final_sequence_length,
                                    self.config.max_position_embeddings)

        # If the context is too big, this happens
        if prompt_length >= final_sequence_length:
            raise ValueError('context length + tokens_to_generate too large')

        # Initialize inference parameters.
        if query_embeds is None:
            self.inference_params = InferenceParams(beam_size,
                                                    final_sequence_length)
        else:
            self.inference_params = InferenceParams(beam_size,
                                                    final_sequence_length + query_embeds.size(1))

        beam_hyp = BeamHypotheses(beam_size)
        done = False
        scores = torch.zeros(
            beam_size, dtype=torch.float32,
            device=torch.cuda.current_device()).unsqueeze(1)

        # =============
        # Run infernece
        # =============
        tokens = tokens.repeat(beam_size, 1)
        if query_embeds is None:
            attention_mask, position_ids = \
                GPT3Model.build_attention_mask_and_position_ids(tokens)
        else:
            query_embeds = query_embeds.repeat(beam_size, 1, 1)
            attention_mask, position_ids = \
                GPT3Model.build_attention_mask_and_position_ids(torch.cat([query_embeds[...,0], tokens.float()], dim=1))

        prev_context_length = 0
        total_prompt_length = prompt_length if query_embeds is None else prompt_length + query_embeds.size(1)
        total_final_sequence_length = final_sequence_length if query_embeds is None else final_sequence_length + query_embeds.size(1)
        for context_length in range(total_prompt_length, total_final_sequence_length):

            # Pick the slice that we need to pass through the network.
            if query_embeds is None:
                tokens2use = tokens[:, prev_context_length:context_length]
            else:
                tokens2use = tokens[:, max(prev_context_length-query_embeds.size(1), 0):context_length-query_embeds.size(1)]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # logits will be meanigful only in the last pipeline stage.
            logits = self(
                tokens=tokens2use, 
                query_embeds=query_embeds if context_length == total_prompt_length else None,
                attention_mask=attention_mask2use, 
                position_ids=positions2use
            ).logits

            vocab_size = logits.size(2)
            log_probs = F.log_softmax(logits, dim=2)
            new_scores = log_probs[:, -1, :] + scores

            if context_length == total_prompt_length:  # if this is the first one
                sorted_scores, indices = torch.sort(
                    new_scores[0, :], descending=True)
            else:
                sorted_scores, indices = torch.sort(
                    new_scores.view(-1), descending=True)

            best_beam_ids = torch.div(indices[:2 * beam_size],
                                      vocab_size*1.0).trunc().long()
            best_words = indices[:2 * beam_size] % vocab_size
            best_scores = sorted_scores[:2 * beam_size]

            next_beams = []
            for beam_token_rank, (token_id, beam_score, beam_id) in enumerate(
                    zip(best_words, best_scores, best_beam_ids)):
                if token_id.item() == stop_token:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= beam_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(tokens[beam_id].clone(), beam_score,
                                 context_length + 1 - total_prompt_length)
                else:
                    # add next predicted token since it is not eos_token
                    next_beams.append((token_id, beam_score, beam_id))

                if len(next_beams) == beam_size:
                    break

            if beam_hyp.is_done(best_scores.max().item(),
                                context_length + 1 - total_prompt_length):
                done = True
                break

            best_batches = tokens.new([item[2] for item in next_beams])
            tokens = tokens[best_batches, :]
            if query_embeds is None:
                tokens[:, context_length] = tokens.new(
                    [item[0] for item in next_beams])
            else:
                tokens[:, context_length-query_embeds.size(1)] = tokens.new(
                    [item[0] for item in next_beams])
            scores = scores.new([item[1] for item in next_beams]).unsqueeze(1)

            # set inference key values to make it consistent with best beam index
            self.inference_params.swap_key_value_dict(best_batches)

            # Update the context length for the next token generation.
            prev_context_length = context_length

        # if cannot find stop token, add open beams to hyps
        if not done:
            for beam_id in range(beam_size):
                beam_hyp.add(tokens[beam_id].clone(), scores[beam_id],
                             context_length + 1 - total_prompt_length)

        # rank based on scores
        sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0], reverse=True)
        num_return_gen = min(num_return_gen, len(sorted_hyps))
        scores = [sorted_hyps[i][0] for i in range(num_return_gen)]
        tokens = [sorted_hyps[i][1] for i in range(num_return_gen)]
        scores = torch.stack(scores, dim=0)
        tokens = torch.stack(tokens, dim=0)

        return addict.Dict(sequences=tokens, scores=scores)

    @torch.no_grad()
    def generate(self, tokens, do_sample=True, termination_id=None, *args, **kwargs):
        if do_sample:
            return self.sample(tokens, termination_id=termination_id, *args, **kwargs)
        else:
            return self.beam_search(tokens, stop_token=termination_id, *args, **kwargs)

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     return self.dist_model.state_dict(destination, prefix, keep_vars)

    def save_pretrained(self,
                        target_folder: Union[str, os.PathLike],
                        save_checkpoint_names: Union[str, List[str]] = None,
                        save_function: Callable = None,
                        config: Optional[dict] = None,
                        **kwargs):
        # DistributedPipeline type is different from task name
        config['pipeline']['type'] = 'gpt3-generation'
        # a temp fix for master_ip, master_port and rank
        # can be removed after refactoring megatron_util
        for unused_key in ('master_ip', 'master_port', 'rank'):
            config['model'].pop(unused_key, None)

        return super().save_pretrained(target_folder, save_checkpoint_names,
                                       save_checkpoint, config, **kwargs)


class BeamHypotheses:

    def __init__(self,
                 num_beams: int,
                 length_penalty: float = 1.0,
                 early_stopping: bool = False):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self,
            hyp: torch.LongTensor,
            sum_logprobs: float,
            beam_indices: Optional[torch.LongTensor] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1]**self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([
                    (s, idx) for idx, (s, _, _) in enumerate(self.beams)
                ])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
