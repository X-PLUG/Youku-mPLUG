""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import json
import torch
from torch import optim as optim

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


from torch.optim import Optimizer

class FusedOptimizer(Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers
        param_groups = []
        for optimizer in self.optimizers:
            param_groups += optimizer.param_groups
        #super(FusedOptimizer, self).__init__([], {})
        self.param_groups = param_groups

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


# def create_optimizer(args, model, filter_bias_and_bn=True):
#     opt_lower = args.opt.lower()
#     weight_decay = args.weight_decay
#     if weight_decay and filter_bias_and_bn:
#         skip = {}
#         if hasattr(model, 'no_weight_decay'):
#             skip = model.no_weight_decay()
#         parameters = add_weight_decay(model, weight_decay, skip)
#         weight_decay = 0.
#     else:
#         parameters = model.parameters()

#     if 'fused' in opt_lower:
#         assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

#     opt_args = dict(lr=args.lr, weight_decay=weight_decay)
#     if hasattr(args, 'opt_eps') and args.opt_eps is not None:
#         opt_args['eps'] = args.opt_eps
#     if hasattr(args, 'opt_betas') and args.opt_betas is not None:
#         opt_args['betas'] = args.opt_betas
#     if hasattr(args, 'opt_args') and args.opt_args is not None:
#         opt_args.update(args.opt_args)

#     opt_split = opt_lower.split('_')
#     opt_lower = opt_split[-1]
#     if opt_lower == 'sgd' or opt_lower == 'nesterov':
#         opt_args.pop('eps', None)
#         optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
#     elif opt_lower == 'momentum':
#         opt_args.pop('eps', None)
#         optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
#     elif opt_lower == 'adam':
#         optimizer = optim.Adam(parameters, **opt_args)
#     elif opt_lower == 'adamw':
#         optimizer = optim.AdamW(parameters, **opt_args)
#     elif opt_lower == 'nadam':
#         optimizer = Nadam(parameters, **opt_args)
#     elif opt_lower == 'radam':
#         optimizer = RAdam(parameters, **opt_args)
#     elif opt_lower == 'adamp':        
#         optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
#     elif opt_lower == 'sgdp':        
#         optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
#     elif opt_lower == 'adadelta':
#         optimizer = optim.Adadelta(parameters, **opt_args)
#     elif opt_lower == 'adafactor':
#         if not args.lr:
#             opt_args['lr'] = None
#         optimizer = Adafactor(parameters, **opt_args)
#     elif opt_lower == 'adahessian':
#         optimizer = Adahessian(parameters, **opt_args)
#     elif opt_lower == 'rmsprop':
#         optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
#     elif opt_lower == 'rmsproptf':
#         optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
#     elif opt_lower == 'novograd':
#         optimizer = NovoGrad(parameters, **opt_args)
#     elif opt_lower == 'nvnovograd':
#         optimizer = NvNovoGrad(parameters, **opt_args)
#     elif opt_lower == 'fusedsgd':
#         opt_args.pop('eps', None)
#         optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
#     elif opt_lower == 'fusedmomentum':
#         opt_args.pop('eps', None)
#         optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
#     elif opt_lower == 'fusedadam':
#         optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
#     elif opt_lower == 'fusedadamw':
#         optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
#     elif opt_lower == 'fusedlamb':
#         optimizer = FusedLAMB(parameters, **opt_args)
#     elif opt_lower == 'fusednovograd':
#         opt_args.setdefault('betas', (0.95, 0.98))
#         optimizer = FusedNovoGrad(parameters, **opt_args)
#     else:
#         assert False and "Invalid optimizer"
#         raise ValueError

#     if len(opt_split) > 1:
#         if opt_split[0] == 'lookahead':
#             optimizer = Lookahead(optimizer)

#     return optimizer


def create_two_optimizer(args, model, filter_bias_and_bn=True):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": args.weight_decay,
            "lr": args.lr1
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": 0.0,
            "lr": args.lr1
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": args.weight_decay,
            "lr": args.lr2
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n))],
            "weight_decay": 0.0,
            "lr": args.lr2
        },

    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    return optimizer


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def create_two_optimizer_video(args, model, filter_bias_and_bn=True):
    no_decay = ["bias", "LayerNorm.weight"]
    new_params = ["lmhra"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n)) and not check_keywords_in_name(n, new_params)],
            "weight_decay": args.weight_decay,
            "lr": args.lr1
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n)) and not check_keywords_in_name(n, new_params)],
            "weight_decay": 0.0,
            "lr": args.lr1
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((not any(nd in n for nd in no_decay)) and ("visual_encoder" not in n)) and check_keywords_in_name(n, new_params)],
            "weight_decay": args.weight_decay,
            "lr": args.lr2
        },
        {
            "params": [p for n, p in model.visual_encoder.named_parameters() if
                       ((any(nd in n for nd in no_decay)) and ("visual_encoder" not in n)) and check_keywords_in_name(n, new_params)],
            "weight_decay": 0.0,
            "lr": args.lr2
        },

    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, betas=args.betas)
    return optimizer




def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, visual_backbone_scale=False):
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

        if visual_backbone_scale and 'visual_encoder.' in name and 'temporal' not in name:
            group_name = "visual_encoder_%s" % (group_name)

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            elif visual_backbone_scale and 'visual_encoder.' in name and 'temporal' not in name:
                scale = 0.1
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


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    
    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer