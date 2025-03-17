from contextlib import asynccontextmanager
import yaml
import uvicorn
import argparse
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from src.dto.inference import InferenceRequest, InferenceResponse
from src.workflows import TrainerPipeline
from src.workflows.inference import InferencePipeline
from src import logger, verify_token
from data import get_file_path

# Function to load YAML configuration
def load_config(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # During startup, instantiate both pipelines using the provided file path.
    file_path = get_file_path()
    app.state.inference_pipeline = InferencePipeline(file_path)
    app.state.trainer_pipeline = TrainerPipeline(file_path)
    yield
    # Shutdown cleanup if needed (e.g., closing DB connections) can be added here.

# Create FastAPI app with Swagger/OpenAPI metadata.
app = FastAPI(
    title="Inference and Training API",
    description="API for running inference and training pipelines.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Inference endpoint secured by token verification.
@app.post("/inference", response_model=InferenceResponse, dependencies=[Depends(verify_token)])
async def inference_endpoint(inference_request: InferenceRequest):
    pipeline: InferencePipeline = app.state.inference_pipeline
    generated_text = pipeline.run(inference_request.context)
    return InferenceResponse(generated_text=generated_text)

# Optional Training endpoint.
@app.post("/train", dependencies=[Depends(verify_token)])
async def train_endpoint(background_tasks: BackgroundTasks):
    pipeline: TrainerPipeline = app.state.trainer_pipeline
    # Run training in the background so the API remains responsive.
    background_tasks.add_task(pipeline.run)
    return {"message": "Training started in background."}

# Health check endpoint.
@app.get("/health", dependencies=[Depends(verify_token)])
def health():
    return {"status": "ok"}

def main():
    parser = argparse.ArgumentParser(
        description="Run training or inference pipelines or start the FastAPI server."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        help="Mode: train or inference. If not provided, the FastAPI server will be started.",
        default=None
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--overrides", nargs="*", help="Override config values (e.g., model.lr=0.01)")
    parser.add_argument("--context", type=str, help="Context string for inference mode", default="Hello, world!")
    args = parser.parse_args()

    # If mode is specified, run the corresponding pipeline directly and exit.
    if args.mode == "train":
        file_path = get_file_path()
        trainer_pipeline = TrainerPipeline(file_path)
        trainer_pipeline.run()
        print("Training completed.")
        return
    elif args.mode == "inference":
        file_path = get_file_path()
        inference_pipeline = InferencePipeline(file_path)
        result = inference_pipeline.run(args.context)
        print("Inference result:", result)
        return
    else:
        # No mode provided: start the FastAPI server.
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
