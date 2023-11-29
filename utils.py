import numpy as np
import io
import sh
import json
import math
import os
import time
from collections import defaultdict, deque
import datetime
import glob
from pathlib import Path

from timm.utils import get_state_dict

import torch
import torch.distributed as dist
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
    

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1, sched_type="cos"):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if sched_type == "cos" or sched_type == "cosine":
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = np.array([
            final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    elif sched_type == "linear":
        schedule = np.linspace(base_value, final_value, epochs * niter_per_ep - warmup_iters)
    else:
        raise NotImplementedError()

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model_iter(args, iter, model, rank):
    output_dir = Path(args.output_dir)
    num_iter = str(iter)
    client_state = {'num_iter': iter}
    model.save_checkpoint(save_dir=args.output_dir, tag="ckpt-iter-%s" % num_iter, client_state=client_state)
    
    if rank == 0:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'ckpt-iter-*'))
        iter = []
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            iter.append(int(t))
        iter.sort()
        if len(iter) > 10:
            rm_ckpt = os.path.join(args.output_dir, 'ckpt-iter-%d' % iter[0])
            sh.rm('-rf', rm_ckpt)


def auto_load_model_iter(args, model):
    output_dir = Path(args.output_dir)
    print(output_dir)
    all_checkpoints = glob.glob(os.path.join(output_dir, 'ckpt-iter-*'))
    iter = []
    for ckpt in all_checkpoints:
        t = ckpt.split('-')[-1].split('.')[0]
        iter.append(int(t))
    iter.sort()
    print("Auto resume iter:", iter[-2])
    _, _ = model.load_checkpoint(args.output_dir, tag='ckpt-iter-%d' % iter[-2])
    model.train()


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:     # torch amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                # args.start_epoch = checkpoint['epoch'] + 1
                setattr(args, "start_epoch", checkpoint['epoch'] + 1)
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                    print('loss scaler', checkpoint['scaler'])
                print("With optim & sched!")
    else:   # ds
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit(): latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                # args.start_epoch = client_states['epoch'] + 1
                setattr(args, "start_epoch", client_states['epoch'] + 1)
                if model_ema is not None:
                    if args.model_ema: _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "ds_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "betas": [args.opt_betas[0], args.opt_betas[1]],
                    "eps": args.opt_eps,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                }
            },
            "fp16": {
                "enabled": not args.bf16,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 500,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": args.bf16
            },
            "amp": {
                "enabled": False,
                "opt_level": "O2"
            },
            "flops_profiler": {
                "enabled": True,
                "profile_step": -1,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
            },
        }

        if args.clip_grad is not None:
            ds_config.update({'gradient_clipping': args.clip_grad})

        if args.zero_stage == 1:
            ds_config.update({"zero_optimization": {"stage": args.zero_stage, "reduce_bucket_size": 5e8}})
        elif args.zero_stage == 2:
            ds_config.update({
                "zero_optimization": {
                    "stage": args.zero_stage,
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e8,
                    "allgather_bucket_size": 5e8
                }
            })
        elif args.zero_stage == 3:
            ds_config.update({
                "zero_optimization": {
                    "stage": args.zero_stage,
                    "contiguous_gradients": True,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_prefetch_bucket_size": 1e7,
                    "stage3_param_persistence_threshold": 1e5,
                    "reduce_bucket_size": 1e7,
                    "sub_group_size": 1e9,
                    "offload_optimizer": {
                        "device": "cpu"
                    },
                    "offload_param": {
                        "device": "cpu"
                    }
                }
            })
            # raise NotImplementedError()

        writer.write(json.dumps(ds_config, indent=2))



############## File Utils ##################

# Copyright (c) Alibaba, Inc. and its affiliates.

import contextlib
import os
import tempfile
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Generator, Union

import requests


class Storage(metaclass=ABCMeta):
    """Abstract class of storage.

    All backends need to implement two apis: ``read()`` and ``read_text()``.
    ``read()`` reads the file as a byte stream and ``read_text()`` reads
    the file as texts.
    """

    @abstractmethod
    def read(self, filepath: str):
        pass

    @abstractmethod
    def read_text(self, filepath: str):
        pass

    @abstractmethod
    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def write_text(self,
                   obj: str,
                   filepath: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        pass


class LocalStorage(Storage):
    """Local hard disk storage"""

    def read(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            content = f.read()
        return content

    def read_text(self,
                  filepath: Union[str, Path],
                  encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``write`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(obj)

    def write_text(self,
                   obj: str,
                   filepath: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``write_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    @contextlib.contextmanager
    def as_local_path(
            self,
            filepath: Union[str,
                            Path]) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        yield filepath


class HTTPStorage(Storage):
    """HTTP and HTTPS storage."""

    def read(self, url):
        # TODO @wenmeng.zwm add progress bar if file is too large
        r = requests.get(url)
        r.raise_for_status()
        return r.content

    def read_text(self, url):
        r = requests.get(url)
        r.raise_for_status()
        return r.text

    @contextlib.contextmanager
    def as_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``as_local_path`` is decorated by :meth:`contextlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> storage = HTTPStorage()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with storage.get_local_path('http://path/to/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.read(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def write(self, obj: bytes, url: Union[str, Path]) -> None:
        raise NotImplementedError('write is not supported by HTTP Storage')

    def write_text(self,
                   obj: str,
                   url: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        raise NotImplementedError(
            'write_text is not supported by HTTP Storage')


class OSSStorage(Storage):
    """OSS storage."""

    def __init__(self, oss_config_file=None):
        # read from config file or env var
        raise NotImplementedError(
            'OSSStorage.__init__ to be implemented in the future')

    def read(self, filepath):
        raise NotImplementedError(
            'OSSStorage.read to be implemented in the future')

    def read_text(self, filepath, encoding='utf-8'):
        raise NotImplementedError(
            'OSSStorage.read_text to be implemented in the future')

    @contextlib.contextmanager
    def as_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``as_local_path`` is decorated by :meth:`contextlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> storage = OSSStorage()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with storage.get_local_path('http://path/to/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.read(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def write(self, obj: bytes, filepath: Union[str, Path]) -> None:
        raise NotImplementedError(
            'OSSStorage.write to be implemented in the future')

    def write_text(self,
                   obj: str,
                   filepath: Union[str, Path],
                   encoding: str = 'utf-8') -> None:
        raise NotImplementedError(
            'OSSStorage.write_text to be implemented in the future')


G_STORAGES = {}


class File(object):
    _prefix_to_storage: dict = {
        'oss': OSSStorage,
        'http': HTTPStorage,
        'https': HTTPStorage,
        'local': LocalStorage,
    }

    @staticmethod
    def _get_storage(uri):
        assert isinstance(uri,
                          str), f'uri should be str type, but got {type(uri)}'

        if '://' not in uri:
            # local path
            storage_type = 'local'
        else:
            prefix, _ = uri.split('://')
            storage_type = prefix

        assert storage_type in File._prefix_to_storage, \
            f'Unsupported uri {uri}, valid prefixs: '\
            f'{list(File._prefix_to_storage.keys())}'

        if storage_type not in G_STORAGES:
            G_STORAGES[storage_type] = File._prefix_to_storage[storage_type]()

        return G_STORAGES[storage_type]

    @staticmethod
    def read(uri: str) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        storage = File._get_storage(uri)
        return storage.read(uri)

    @staticmethod
    def read_text(uri: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        storage = File._get_storage(uri)
        return storage.read_text(uri)

    @staticmethod
    def write(obj: bytes, uri: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``write`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        storage = File._get_storage(uri)
        return storage.write(obj, uri)

    @staticmethod
    def write_text(obj: str, uri: str, encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``write_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        storage = File._get_storage(uri)
        return storage.write_text(obj, uri)

    @contextlib.contextmanager
    def as_local_path(uri: str) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing."""
        storage = File._get_storage(uri)
        with storage.as_local_path(uri) as local_path:
            yield local_path
