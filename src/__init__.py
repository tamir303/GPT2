from src.model import GPT2
from src.config import Config
from src.eval import estimate_loss as evaluation
from src.utils import get_batch, split_train_test
from src.tokenizer import Tokenizer