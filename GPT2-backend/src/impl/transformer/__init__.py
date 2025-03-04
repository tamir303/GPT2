from torch import nn
from src.impl.transformer.feedforward import FFN
from src.impl.transformer.multihead import MultiHeadAttention
from src.impl.transformer.block import DecoderBlock, EncoderBlock, DecoderOutputBlock


def get_transformer_encoder_block(n_layers, n_heads, d_model, d_ff, dropout, block_size):
    return nn.Sequential(
        *[EncoderBlock(n_heads, d_ff, d_model, dropout, block_size) for _ in range(n_layers)]
    )


def get_transformer_decoder_block(n_layers, n_heads, d_model, d_ff, dropout, block_size):
    return nn.Sequential(
        *[DecoderBlock(n_heads, d_ff, d_model, dropout, block_size) for _ in range(n_layers)]
    )