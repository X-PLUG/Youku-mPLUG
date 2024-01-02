# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron initialization."""

import os
import random
import time
from typing import Dict

import numpy as np
import torch
from datetime import timedelta

from megatron_util import fused_kernels
from megatron_util import get_args
from megatron_util import mpu
from megatron_util.global_vars import set_global_variables
from megatron_util import mpu
from megatron_util.mpu import (set_tensor_model_parallel_rank,
                          set_tensor_model_parallel_world_size)


def initialize_megatron(megatron_cfg: Dict):
    assert torch.cuda.is_available(), 'Megatron util requires CUDA.'

    version = megatron_cfg.pop('version', 'v3')
    if version == 'v1':
        _initialize_v1()
    elif version == 'moe':
        _initialize_moe()

    set_global_variables(megatron_cfg)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()
        
        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    args = get_args()
    if  args.lazy_mpu_init:
        args.use_cpu_initialization=True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals    
        set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Compile dependencies.
        _compile_dependencies()

        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # ==================
    # Load fused kernels
    # ==================

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling and loading fused kernels ...', flush=True)
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(
                  time.time() - start_time), flush=True)


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if 'local_rank' in args:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
        # Call the init process
        master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
        master_port = int(os.getenv('MASTER_PORT', '29500'))
        init_method = f'tcp://{master_addr}:{master_port}'
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method,
            timeout=timedelta(minutes=3000))

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            mpu.initialize_model_parallel(args.tensor_model_parallel_size,
                                          args.pipeline_model_parallel_size,
                                          args.virtual_pipeline_model_parallel_size,
                                          args.pipeline_model_parallel_split_rank)


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def _initialize_v1():
    from megatron_util.mpu.layers_v1 import (VocabParallelEmbedding,
                                             ParallelEmbedding,
                                             ColumnParallelLinear,
                                             RowParallelLinear)
    mpu.VocabParallelEmbedding = VocabParallelEmbedding
    mpu.ParallelEmbedding = ParallelEmbedding
    mpu.ColumnParallelLinear = ColumnParallelLinear
    mpu.RowParallelLinear = RowParallelLinear

    mpu.copy_to_model_parallel_region = mpu.copy_to_tensor_model_parallel_region
    mpu.reduce_from_model_parallel_region = mpu.reduce_from_tensor_model_parallel_region
    mpu.scatter_to_model_parallel_region = mpu.scatter_to_tensor_model_parallel_region
    mpu.gather_from_model_parallel_region = mpu.gather_from_tensor_model_parallel_region

    mpu.get_model_parallel_group = mpu.get_tensor_model_parallel_group
    mpu.get_model_parallel_world_size = mpu.get_tensor_model_parallel_world_size
    mpu.get_model_parallel_rank = mpu.get_tensor_model_parallel_rank
    mpu.get_model_parallel_src_rank = mpu.get_tensor_model_parallel_src_rank


def _initialize_moe():
    from megatron_util.mpu.layers_moe import ColumnParallelLinear, RowParallelLinear
    mpu.ColumnParallelLinear = ColumnParallelLinear
    mpu.RowParallelLinear = RowParallelLinear
