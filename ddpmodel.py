import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import torch
from torch import nn
from torch.nn import functional as F
import math
import os
import glob
import random
from functools import partial
import sys
sys.path.append('./src/')
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from contextlib import nullcontext
import matplotlib.pyplot as plt
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer
from data_loader_ddp import *
from utils import *
from model import *
from torch.profiler import profile, record_function, ProfilerActivity
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

#     ds = PretokDataset(data_cache_dir=DATA_CACHE_DIR, split="train", max_seq_len=max_seq_len, rank=rank, world_size=world_size)
#     dl = DataLoader(ds, batch_size=4, pin_memory=True, num_workers=0)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    
    print(f"Running DDP on rank {rank}.")
    
    # mixed precision settings
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device="cuda"
    dtype = 'bfloat16'
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device, dtype=ptdtype)
    )
    best_val_loss = float('inf')
    
    # load dataset
    DATA_CACHE_DIR = './pangenome_dataset/'
    out_dir = './models-7b_test_DDP/'
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/scratch/ac.zyang/LLM/src/llama2/pangenome-tokenizer")
    
    # load model
    model_args = ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=4,
        vocab_size=32000,
        multiple_of=256,
        max_seq_len=350,
        dropout=0.0,
    )
    model = Transformer(model_args).to(rank)
    model = DDP(model, device_ids=[rank])
    print(f'Rank:{rank}, Number of parameters: {sum(p.nelement() for p in model.parameters())}')
    
    # training hyper parameters
    batch_size = 1
    wanted_batch_size = 4 * 128
    gradient_accumulation_steps = wanted_batch_size // batch_size
    max_iters = 250
    eval_iters = 5
    best_val_loss = 1e9
    grad_clip = 1
    max_seq_len=350
    
    learning_rate = 5e-4
    optimizer = get_optimizer(
        model=model,
        device_type='cuda',
        learning_rate=learning_rate,  # max learning rate
        weight_decay = 1e-1,
        beta1 = 0.9,
        beta2 = 0.95,
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    
    ds = PretokDataset(max_seq_len=350,split="train", rank=rank, data_cache_dir=DATA_CACHE_DIR)
    dl = torch.utils.data.DataLoader(
        dataset=ds, 
        batch_size=batch_size,
        pin_memory=True, num_workers=0
    )
    train_loss, val_loss = [], []
    
    # Start training
    iter_num = 0
    for X, Y in dl:
        X = X.to(rank)
        Y = Y.to(rank)

        lr = get_lr(iter_num, max_iters=max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

            
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits = model(X)
                loss = compute_loss(logits, Y)
                loss = loss / gradient_accumulation_steps  
                
            print(f"Rank {rank}, Iteration {iter_num}, Loss: {loss.item()}")
            scaler.scale(loss).backward()
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


        iter_num += 1
        if iter_num > max_iters:
            break

    if rank == 0:
        torch.save(model.state_dict(), "trained_model.pth")

    cleanup()


