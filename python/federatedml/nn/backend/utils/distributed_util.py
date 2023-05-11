import torch.distributed as dist


def is_rank_0():
    return not dist.is_available() or dist.get_rank() == 0


def is_distributed():
    return dist.is_available()


def get_num_workers():
    return dist.get_world_size()
