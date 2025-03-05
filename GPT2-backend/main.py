import uvicorn

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from src.impl.model_ops import get_info, setup_tokenizer
from src.concrete.data import DataTokenizer, DataIngestionStep, DataSplitStep
from src.concrete.model import ModelCreateStep, ModelTrainStep, ModelEvaluateStep, ModelInferenceStep
from src.concrete.load import LoadModelStep, LoadTokenizerStep
from src.concrete.input import UserInputStep
from src.concrete.pipeline import Pipeline

from model.inference_model import InferenceRequest
from model.tokenizer_model import TokenizerEncodeRequest, TokenizerDecodeRequest, TokenizerDecodeResponse, TokenizerEncodeResponse

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


### Info API ###
@app.get("/health", tags=["Root"])
async def health_check():
    return {"status": "ok"}

@app.get("/info", tags=["Root"])
async def info():
    return JSONResponse(get_info())


### Training API ###
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


### Inference API ###
@app.post("/inference", tags=["Inference"])
async def inference(request: InferenceRequest):
    pipeline = pipelines["inference"]
    generated_text = pipeline.run(request.prompt)
    return JSONResponse(content={"generated_text": generated_text})


### Tokenizer API ###
@app.post("/tokenizer/encode", tags=["Tokenizer"], response_model=TokenizerEncodeResponse)
async def tokenizer_encode(request: TokenizerEncodeRequest):
    tokenizer = setup_tokenizer()
    encoded_list = tokenizer.encode(request.raw).tolist()[0]
    return TokenizerEncodeResponse(encoded_text=request.raw, encoded_ids=encoded_list)

@app.post("/tokenizer/decode", tags=["Tokenizer"], response_model=TokenizerDecodeResponse)
async def tokenizer_decode(request: TokenizerDecodeRequest):
    tokenizer = setup_tokenizer()
    decoded = tokenizer.decode(request.token_ids)
    return TokenizerDecodeResponse(decoded_text=decoded, decoded_tokens=request.token_ids)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
