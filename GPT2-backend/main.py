import uvicorn

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from src.impl.model_ops import get_info
from src.concrete.data import DataTokenizer, DataIngestionStep, DataSplitStep
from src.concrete.model import ModelCreateStep, ModelTrainStep, ModelEvaluateStep, ModelInferenceStep
from src.concrete.load import LoadModelStep, LoadTokenizerStep
from src.concrete.input import UserInputStep
from src.concrete.pipeline import Pipeline

from model.inference_model import InferenceRequest

pipelines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipelines["new_train"] = Pipeline(steps=[
            DataIngestionStep(source_path="data/ClimateChangeAnalysis.txt"),
            DataSplitStep(),
            DataTokenizer(vocab_file="tok400.vocab"),
            ModelCreateStep(),
            ModelTrainStep(),
            ModelEvaluateStep()
        ])

        pipelines["re_train"] = Pipeline(steps=[
            DataIngestionStep(source_path="data/ClimateChangeAnalysis.txt"),
            DataSplitStep(),
            LoadTokenizerStep(vocab_file="tok400.vocab"),
            LoadModelStep(),
            ModelTrainStep(),
            ModelEvaluateStep()
        ])

        pipelines["inference"] = Pipeline(steps=[
            UserInputStep(),
            LoadTokenizerStep(vocab_file="tok400.vocab"),
            LoadModelStep(),
            ModelInferenceStep()
        ])
        yield
    finally:
        pipelines.clear()

app = FastAPI(
    title="Transformer Model API",
    description="Transformer Model API",
    lifespan=lifespan,
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc"
)

@app.get("/", tags=["Root"])
async def main():
    return RedirectResponse(url="/info")


@app.get("/health", tags=["Root"])
async def health_check():
    return {"status": "ok"}


@app.get("/info", tags=["Root"])
async def info():
    return JSONResponse(get_info())


@app.get("/train/new", tags=["Train"])
async def train(background_tasks: BackgroundTasks):
    pipeline = pipelines["new_train"]
    background_tasks.add_task(pipeline.run())
    return { "message": "Training started in background" }


@app.get("/train/re", tags=["Train"])
async def train(background_tasks: BackgroundTasks):
    pipeline = pipelines["re_train"]
    background_tasks.add_task(pipeline.run())
    return { "message": "Training started in background" }


@app.post("/inference", tags=["Inference"])
async def inference(request: InferenceRequest):
    pipeline = pipelines["inference"]
    generated_text = pipeline.run(request.prompt)
    return JSONResponse(content={"generated_text": generated_text})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
