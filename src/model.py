import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm

class GPT2(nn.Module):
    def __init__(self,
                vocab_size,
                max_seq_len,
                d_model,
                num_heads,
                num_layers,
                batch_size,
                dropout=0.1):
        super().__init__()
