import torch.distributed as dist


def is_rank_0():
    return dist.get_rank() == 0


def is_distributed():
    return dist.is_initialized()


def get_num_workers():
    return dist.get_world_size()
