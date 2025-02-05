import uvicorn
from fastapi import FastAPI, HTTPException
from src import ModelManager
from api import GenerationRequest, GenerationResponse, load_data
from contextlib import asynccontextmanager

manager = ModelManager()

@asynccontextmanager
async def lifespan(f_app: FastAPI):
    try:
        data = load_data()
        manager.load_model(data)
        yield
        manager.end_experiment()
    except Exception as e:
        # In a production system, use a proper logger instead of print.
        print(f"Error during startup model loading: {e}")

app = FastAPI(
    title="Transformer Model API",
    description="Transformer Model API",
    lifespan=lifespan,
    version="1.0.0",
    docs_url="/swagger",  # Changes the Swagger UI path
    redoc_url="/redoc"  # Optionally, change the ReDoc path
)

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.post("/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate_text(request: GenerationRequest):
    try:
        generated = manager.generate_text(request.prompt, request.max_new_tokens)
        return GenerationResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)