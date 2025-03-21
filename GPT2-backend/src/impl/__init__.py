import torch
from torch.optim.adamw import AdamW

from src.etc.config import HyperParams, Config
from src.impl.model.model import GPT2, IModel
from src.impl.tokenizer.tokenizer import ITokenizer, Tokenizer
from src.impl.training.trainer import ITraining, Trainer
from src.impl.evaluation.loss import estimate_loss
from src.impl.data.dataloader import DataLoader, IDataLoader
from src.impl.utils.split_batch_utils import split_train_test


def init_model(vocab_size) -> IModel:
    return GPT2(vocab_size, Config.block_size, Config.d_model, Config.n_heads, Config.n_layers, Config.dropout)

def init_tokenizer(file_path: str) -> ITokenizer:
    return Tokenizer(file_path)

def init_trainer(model: IModel, optimizer: AdamW) -> ITraining:
    return Trainer(model, optimizer)

def get_loss(model: IModel, data: torch.Tensor) -> dict:
    return estimate_loss(model, data)

def initialize_optimizer(model, config: HyperParams) -> any:
    model.to(Config.device)

    if config.optimizer_type == 'AdamW':
        return AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.epsilon),
            foreach=True
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

def split_data(data: torch.Tensor, split: float):
    return split_train_test(data, split)

def get_data_loader(file_path: str) -> IDataLoader:
    return DataLoader(file_path)