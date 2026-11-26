"""Construct split-specific loaders and initialize worker random states."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .ubfc_dataset import UBFCChunks

def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

def make_loader(cfg, split, generator=None):
    dataset = UBFCChunks(cfg, split)
    return DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=split == "train",
                      num_workers=cfg["train"]["num_workers"], drop_last=False,
                      pin_memory=cfg["train"]["device"].startswith("cuda"),
                      worker_init_fn=worker_seed, generator=generator)
