from dataclasses import dataclass
from model import PashkoModel
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import tiktoken
import math
import numpy as np
import csv
from data_chunks import load_dataset

"""Config and init"""

@dataclass
class PashkoModelConfig:
    sequence_length: int = 1024
    vocab_size: int = 50304
    embed_dim: int = 768
    encoder = 'gpt2'
    num_heads: int = 12
    num_blocks: int = 12
    dropout: float = 0.0
    ffnn_bias: bool = False
    qkv_bias: bool = False
    layernorm_bias: bool = False
    topK: int = 10
    temperature: float = 1.0

@dataclass
class PashkoTrainConfig:
    batch_size: int = 64
    learning_rate: float = 6e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 1e-1
    decay_learning_rate = bool = True
    warmup_iters: int = 2000
    min_learning_rate = float = 6e-5
    gradient_clip = 1.0
    max_iterations: int = 600000
    evaluation_interval: int = 2000
    log_interval: int = 10
    checkpoint_interval: int = 5000

device = "cuda" if torch.cuda.is_available() else "cpu"

modelConfig = PashkoModelConfig()
trainConfig = PashkoTrainConfig()
iter_num = 0
ckpt_num = 1
best_val_loss = 1e9
mfu = 0.49509986693212665

print("Preparing data")

"""Data Preparation"""

num_chunks = 1012
chunk_shape = (32768,modelConfig.sequence_length)

all_chunks = npy_files = [f for f in os.listdir('dataset_chunks') if f.endswith(".npy")]

data_split = round(num_chunks * 0.9)

train_chunks = all_chunks[data_split:]

"""TODO Dataloaders for training loop"""


print("Data train/val split")

"""Model and Optimiser Initialisation"""

print("Type of initialisation...", end=' ', flush=True)
init_type = input()
print(init_type)

if init_type == "scratch":
    Pashko = PashkoModel(modelConfig)
    Pashko = Pashko.to(device)
    print(f"Initialised new model with {Pashko.num_params()[0]} parameters")

    optimiser = torch.optim.AdamW(Pashko.parameters(),
                              lr=trainConfig.learning_rate,
                              betas=trainConfig.betas,
                              weight_decay=trainConfig.weight_decay)
    
    print("Initialised optimiser using trainConfig")

    checkpoint = {
                    'model': Pashko.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'iter_num': iter_num,
                    'ckpt_num': ckpt_num,
                    'best_val_loss': best_val_loss,
                    'model_config': modelConfig,
                    'train_config': trainConfig,
                }
    
    torch.save(checkpoint, os.path.join('checkpoints', f'ckpt0.pashko'))

    print(f"Saved initial checkpoint as ckpt{ckpt_num}.pt")

    print("Initialisation complete")

elif init_type == "resume":
    print("Checkpoint name...", end=' ', flush=True)
    ckpt_name = input()
    print(ckpt_name)

    ckpt_path = os.path.join('checkpoints', f'{ckpt_name}.pashko')
    ckpt = torch.load(ckpt_path, map_location=device)

    modelConfig = ckpt['model_config']
    trainConfig = ckpt['train_config']

    Pashko = PashkoModel(modelConfig)
    Pashko.load_state_dict(ckpt['model'])

    Pashko = Pashko.to(device)

    print(f"Loaded model from checkpoint {ckpt_name}")

    optimiser = torch.optim.AdamW(Pashko.parameters(),
                              lr=trainConfig.learning_rate,
                              betas=trainConfig.betas,
                              weight_decay=trainConfig.weight_decay)
    
    optimiser.load_state_dict(ckpt['optimiser'])

    print(f"Loaded optimiser from checkpoint {ckpt_name}")

    iter_num = ckpt['iter_num']
    ckpt_num = ckpt['ckpt_num']
    best_val_loss = ckpt['best_val_loss']

    print("Initialisation complete")


print("Compiling the model...")
Pashko = torch.compile(Pashko)

"""
Training Loop
"""

val_metrics = {
    'iter_num': [],
    'val_loss': []
}

log_metrics = {
    'iter_num': [],
    'train_loss': [],
    'mfu': [],
    'eta': []
}

def learning_rate_at(iter):
    if iter < trainConfig.warmup_iters:
        return trainConfig.learning_rate * iter / trainConfig.warmup_iters
    if iter > trainConfig.max_iters:
        return trainConfig.min_learning_rate
    decay_ratio = (iter - trainConfig.warmup_iters) / (trainConfig.max_iters - trainConfig.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return trainConfig.min_learning_rate + coeff * (trainConfig.learning_rate - trainConfig.min_learning_rate)

print("Beginning training")

t0 = time.time()
while True:
    if trainConfig.decay_learning_rate:
        iter_learning_rate = learning_rate_at(iter_num)
        for param_group in optimiser.param_groups:
            param_group['lr'] = iter_learning_rate

    if iter_num % trainConfig.evalulation_interval == 0:
        val_loss = torch.Tensor([0.])
        
        for batch in valLoader:
            x_batch = batch[0].to(device)
            y_batch = torch.cat([x_batch[:, 1:], -1 * torch.ones_like(x_batch[:, :1])], dim=1).to(device)
            
            logits, loss = Pashko(x_batch,y_batch)
            val_loss += loss
        val_loss = val_loss.item() / num_val_batches

        val_metrics['iter_num'].append(iter_num)
        val_metrics['val_loss'].append(val_loss.item())

        if val_loss < best_val_loss:
            checkpoint = {
                    'model': Pashko.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'iter_num': iter_num,
                    'ckpt_num': ckpt_num,
                    'best_val_loss': val_loss,
                    'model_config': modelConfig,
                    'train_config': trainConfig,
                }
            print("Saving checkpoint best val loss.")
            torch.save(checkpoint, os.path.join('checkpoints', 'ckpt-val.pashko'))

    for batch in trainLoader:
        x_batch = batch[0].to(device)
        y_batch = torch.cat([x_batch[:, 1:], -1 * torch.ones_like(x_batch[:, :1])], dim=1).to(device)

        logits, loss = Pashko(x_batch, y_batch)

        loss.backward()

        if trainConfig.gradient_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(Pashko.parameters(), trainConfig.gradient_clip)

        optimiser.step()

        optimiser.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % trainConfig.log_interval == 0:
        loss = loss.item()
        if iter_num >= 5:
            running_mfu = mfu
            mfu = ((modelConfig.batchSize / (dt/num_train_batches)) * 875336564736 / 136e12)
            mfu = running_mfu*0.9 + mfu*0.1

            eta = ((6 * Pashko.num_params()[1] * 60000000000)  / (136e12 * mfu)) * (1 - iter_num/trainConfig.max_iterations)

            log_metrics['iter_num'].append(iter_num)
            log_metrics['train_loss'].append(loss)
            log_metrics['mfu'].append(mfu*100)
            log_metrics['eta'].append(eta/3600)

        print(f"iter {iter_num}: loss {loss:.4f}, time: {dt*1000:.2f}ms, mfu: {mfu*100:.2f}%, eta: {eta/3600:.2f}h")
        
    iter_num += 1

    if iter_num % trainConfig.checkpoint_interval == 0:
        checkpoint = {
                    'model': Pashko.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'iter_num': iter_num,
                    'ckpt_num': ckpt_num,
                    'best_val_loss': val_loss,
                    'model_config': modelConfig,
                    'train_config': trainConfig,
                }
        print("Saving checkpoint best val loss.")
        torch.save(checkpoint, os.path.join('checkpoints', f'ckpt{ckpt_num}.pashko'))

        val_output_file = f'logs/val_metrics_{ckpt_num}.csv'
        log_output_file = f'logs/log_metrics_{ckpt_num}.csv'

        f = open(val_output_file, "x")
        f = open(log_output_file, "x")

        with open(val_output_file, 'w', newline='') as csvfile:
            fieldnames = list(val_metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(val_metrics['iter_num'])):
                row = {key: val_metrics[key][i] for key in fieldnames}
                writer.writerow(row)

        with open(log_output_file, 'w', newline='') as csvfile:
            fieldnames = list(log_metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(log_metrics['iter_num'])):
                row = {key: log_metrics[key][i] for key in fieldnames}
                writer.writerow(row)
            
    if iter_num > trainConfig.max_iterations:
        break