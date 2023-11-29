import torch
import torch.distributed as dist
from utils import get_rank, is_dist_avail_and_initialized, is_main_process
import random
import logging

logger = logging.getLogger(__name__)


class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, name2loader):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.
        loaders: Dict, {name: dataloader}
        """
        self.name2loader = name2loader
        self.name2iter = {name: iter(l) for name, l in name2loader.items()}
        name2index = {name: idx for idx, (name, l) in enumerate(name2loader.items())}
        index2name = {v: k for k, v in name2index.items()}

        iter_order = []
        for n, l in name2loader.items():
            iter_order.extend([name2index[n]]*len(l))

        random.shuffle(iter_order)
        iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)

        # sync
        if is_dist_avail_and_initialized():
            # make sure all processes have the same order so that
            # each step they will have data from the same loader
            dist.broadcast(iter_order, src=0)
        self.iter_order = [index2name[int(e.item())] for e in iter_order.cpu()]

        logger.info(str(self))

    def __str__(self):
        output = [f"MetaLoader has {len(self.name2loader)} dataloaders, {len(self)} batches in total"]
        for idx, (name, loader) in enumerate(self.name2loader.items()):
            output.append(
                f"dataloader index={idx} name={name}, batch-size={loader.batch_size} length(#batches)={len(loader)} "
            )
        return "\n".join(output)

    def __len__(self):
        return len(self.iter_order)

    def __iter__(self):
        """ this iterator will run indefinitely """
        for name in self.iter_order:
            _iter = self.name2iter[name]
            batch = next(_iter)
            yield name, batch


class MetaLoaderJoint(object):
    """ wraps multiple data loader """

    def __init__(self, name2loader):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.
        loaders: Dict, {name: dataloader}
        """
        self.name2loader = name2loader
        self.name2iter = {
            "image": {name: iter(l) for name, l in name2loader.items() if "image" in name},
            "video": {name: iter(l) for name, l in name2loader.items() if "video" in name}
        }

        name2index = {
            "image": {name: idx for idx, (name, l) in enumerate(name2loader.items()) if "image" in name},
            "video": {name: idx for idx, (name, l) in enumerate(name2loader.items()) if "video" in name}
        }

        index2name = {
            "image": {v: k for k, v in name2index["image"].items()},
            "video": {v: k for k, v in name2index["video"].items()},
        }

        iter_order = {"image": [], "video": []}

        for n, l in name2loader.items():
            data_type = n.split("_")[0]
            iter_order[data_type].extend([name2index[data_type][n]] * len(l))

        self.max_iter_order_len = max([len(v) for k, v in iter_order.items()])

        self.iter_order = {}
        for data_type in ["image", "video"]:

            if len(iter_order[data_type]) < self.max_iter_order_len:
                rest_number = self.max_iter_order_len - len(iter_order[data_type])
                extra = random.choices(iter_order[data_type], k=rest_number)
                iter_order[data_type].extend(extra)

            random.shuffle(iter_order[data_type])
            iter_order_tmp = torch.Tensor(iter_order[data_type]).to(torch.device("cuda")).to(torch.uint8)

            # sync
            if is_dist_avail_and_initialized():
                # make sure all processes have the same order so that
                # each step they will have data from the same loader
                dist.broadcast(iter_order_tmp, src=0)

            self.iter_order[data_type] = [index2name[data_type][int(e.item())] for e in iter_order_tmp.cpu()]

        self.dummy_order = list(range(self.max_iter_order_len))

    def __str__(self):
        output = [f"MetaLoader has {len(self.name2loader)} dataloaders, {len(self)} batches in total"]
        for idx, (name, loader) in enumerate(self.name2loader.items()):
            output.append(
                f"Dataloader index={idx} name={name}, batch-size={loader.batch_size} length(#batches)={len(loader)} "
            )

        return "\n".join(output)

    def __len__(self):
        return self.max_iter_order_len

    def __iter__(self):
        """ this iterator will run indefinitely """
        for idx in self.dummy_order:
            batch = {"image": None, "video": None}
            for data_type in ["image", "video"]:
                name = self.iter_order[data_type][idx]
                try:
                    _iter = self.name2iter[data_type][name]
                    batch[data_type] = next(_iter)
                except StopIteration:
                    print("restart")
                    self.name2iter[data_type][name] = iter(self.name2loader[name])
                    _iter = self.name2iter[data_type][name]
                    batch[data_type] = next(_iter)

            yield batch["image"], batch["video"]