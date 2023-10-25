import torch.distributed as dist
import torch
import os

def cleanup():
    dist.destroy_process_group()
    
def setup(rank, world_size):
    # equivalent to MPI init.
    dist.init_process_group("nccl", init_method="env://",
        world_size=world_size,
        rank=rank)

    # lookup number of ranks in the job, and our rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    ngpus_per_process = torch.cuda.device_count()
    local_rank = rank % ngpus_per_process
    
    print("Setup nodes for training ...")
    print(f"world_size: {world_size}, rank: {rank}, ngpus_per_process: {ngpus_per_process}, local_rank: {local_rank}")
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    return local_rank

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):

    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0