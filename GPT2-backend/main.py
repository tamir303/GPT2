import yaml
import uvicorn
import argparse
from fastapi import Depends, Request
from data import get_file_path
from src.workflows import TrainerPipeline
from src import app, logger, verify_token

# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Inference Endpoint
@app.get("/inference", dependencies=[Depends(verify_token)])
@app.state.limiter.limit("100/minute")
async def inference_endpoint(request: Request):
    return {"message": "Inference result"}

# Training Function
def run_training(cfg):
    logger.info(
        f"\nExperiment:\n{cfg.get('experiment',{})}"
        f"\nModel:\n{cfg.get('model',{})}"
        f"\nTokenizer:\n{cfg.get('tokenizer',{})}"
        f"\nData:\n{cfg.get('data',{})}"
        f"\nTraining:\n{cfg.get('training',{})}"
        f"\nLogging:\n{cfg.get('logging',{})}"
        f"\nMLOps:\n{cfg.get('mlops',{})}"
    )

    file_path = get_file_path()
    trainer_pipeline = TrainerPipeline(file_path)
    trainer_pipeline.run()

# Main Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run inference")
    parser.add_argument("mode", choices=["train", "inference"], help="Mode: train or inference", default="inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--overrides", nargs="*", help="Override config values (e.g., model.lr=0.01)")

    args = parser.parse_args()

    if args.mode == "train":
        # Load configuration from YAML
        cfg = load_config(args.config)

        # Apply overrides (manual parsing)
        if args.overrides:
            for override in args.overrides:
                key, value = override.split("=")
                keys = key.split(".")
                temp = cfg
                for k in keys[:-1]:
                    temp = temp.setdefault(k, {})
                temp[keys[-1]] = value

        run_training(cfg)

    elif args.mode == "inference":
        uvicorn.run(app, host="0.0.0.0", port=8000)
