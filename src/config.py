from dataclasses import dataclass
import torch

@dataclass
class Config:
    # The device to run the src on: 'cuda' if a GPU is available, 'cpu' otherwise.
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Batch size for training. Defines how many samples will be processed together in one step.
    batch_size: int = 16

    # The size of the input sequence that will be fed into the src (for transformer models, this is the block size).
    block_size: int = 32

    # Maximum number of training iterations.
    max_iters: int = 5000

    # Interval between evaluations, i.e., how frequently the src will be evaluated on the validation set during training.
    eval_interval: int = 100

    # Learning rate for the optimizer. Controls how big a step the optimizer takes when updating the src parameters.
    learning_rate: float = 1e-3

    # Number of iterations over the validation set used for evaluation.
    eval_iters: int = 200

    # Dimensionality of the src's hidden representations. Typically, this is the size of the vectors inside the src.
    d_model: int = 64

    # Number of attention heads in the multi-head attention mechanism of the transformer src.
    # More heads allow the src to focus on different parts of the input simultaneously.
    n_heads: int = 4

    # The number of layers in the transformer src. More layers allow for more complex models but increase computation.
    n_layers: int = 4

    # Dropout rate used to prevent overfitting. Dropout randomly drops units from the neural network during training.
    dropout: float = 0.0


torch.manual_seed(1337)