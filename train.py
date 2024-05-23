from ddpmodel import *
import torch

if __name__ == "__main__":
    world_size = 4  # Assuming 4 GPUs
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)