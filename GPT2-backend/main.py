import sys
from fastapi import FastAPI, Depends, Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import warnings
from src.etc.logger import CustomLogger, logging
from src.workflows import TrainerPipeline
from src.middlewares.auth import OAuth2PasswordBearer, verify_token
from src.middlewares.metrics import handle_metrics, PrometheusMiddleware
from src.middlewares.cors import CORSMiddleware
from src.middlewares.limiter import limiter as FLimiter, RateLimitExceeded, _rate_limit_exceeded_handler
from data import get_file_path

import uvicorn

warnings.filterwarnings("ignore")

# Logger setup
logger = CustomLogger(
    log_name='Main',
    log_level=logging.DEBUG,
    log_dir='logs',
    log_filename="runs.log",
).get_logger()

# FastAPI app setup
app = FastAPI()

# Rate Limiting Middleware
limiter = FLimiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication Middleware
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error Handling Middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"ValueError: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# Metrics Middleware
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

# Inference Endpoint
@app.get("/inference", dependencies=[Depends(verify_token)])
@limiter.limit("100/minute")
async def inference_endpoint():
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
        # Use Hydra to load configuration and run training
        with initialize(config_path="."):
            cfg = compose(config_name="config", overrides=sys.argv[2:])
            run_training(cfg)
    elif mode == "inference":
        # Start FastAPI server with middlewares
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Invalid mode. Use 'train' or 'inference'.")