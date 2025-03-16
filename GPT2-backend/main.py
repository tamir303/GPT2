import sys
import uvicorn
from fastapi import Depends, Request
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from data import get_file_path
from src.workflows import TrainerPipeline
from src import app, logger, verify_token


# Inference Endpoint
@app.get("/inference", dependencies=[Depends(verify_token)])
@app.state.limiter.limit("100/minute")
async def inference_endpoint(request: Request):
    return {"message": "Inference result"}


# Training Function
def run_training(cfg: DictConfig):
    logger.info(
        f"\nExperiment:\n{OmegaConf.to_yaml(cfg.experiment)}"
        f"\nModel:\n{OmegaConf.to_yaml(cfg.model)}"
        f"\nTokenizer:\n{OmegaConf.to_yaml(cfg.tokenizer)}"
        f"\nData:\n{OmegaConf.to_yaml(cfg.data)}"
        f"\nTraining:\n{OmegaConf.to_yaml(cfg.training)}"
        f"\nLogging:\n{OmegaConf.to_yaml(cfg.logging)}"
        f"\nmlops:\n{OmegaConf.to_yaml(cfg.mlops)}"
    )

    file_path = get_file_path()
    trainer_pipeline = TrainerPipeline(file_path)
    trainer_pipeline.run()


# Main Entry Point
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inference"

    if mode == "train":
        # Load configuration and start training
        with initialize(config_path="."):
            cfg = compose(config_name="config", overrides=sys.argv[2:])
            run_training(cfg)
    elif mode == "inference":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Invalid mode. Use 'train' or 'inference'.")
