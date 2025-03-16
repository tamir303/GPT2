from dataclasses import dataclass
import torch
import yaml

@dataclass
class HyperParams:
    # Experiment Configuration
    name: str
    version: str
    seed: int

    # Data Configuration
    path: str
    version: str
    split_ratio: float

    # Model Configuration
    type: str
    d_model: int
    block_size: int
    n_heads: int
    n_layers: int
    dropout: float

    # Training Configuration
    max_iters: int
    batch_size: int
    eval_interval: int
    eval_iters: int
    save_interval: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    max_grad_norm: float
    patience: int
    device: str
    load_existing_model: bool
    checkpoint_dir: str
    optimizer_type: str
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float

    # Logging and Monitoring
    log_dir: str
    log_level: str
    log_debug_activate: bool
    mlflow_tracking_uri: str
    mlflow_enabled: bool

    # MLOps Configuration
    distributed: bool
    num_workers: int
    docker_image: str
    artifact_store: str

    # Tokenizer Configuration
    type: str
    vocab_file: str

def load_config(config_path: str) -> HyperParams:
    """Load configuration from YAML file into HyperParams dataclass, ignoring section prefixes."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Flatten nested dictionary, keeping only innermost keys
    flat_config = {}
    for section, values in config_dict.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):  # Handle nested 'mlflow'
                    for sub_key, sub_value in value.items():
                        flat_config[sub_key] = sub_value
                else:
                    flat_config[key] = value
        else:
            flat_config[section] = values


    return HyperParams(**flat_config)

Config = load_config("config.yaml")
Config.device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(Config.seed)
torch.cuda.manual_seed_all(Config.seed)
