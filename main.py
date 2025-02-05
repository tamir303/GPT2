import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from src import ModelManager
from api import GenerationRequest, GenerationResponse, data_sample_test
from api import DocumentProcessor
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import os
from dotenv import load_dotenv

load_dotenv()
manager = ModelManager()
processor : DocumentProcessor | None = None

@asynccontextmanager
async def lifespan(f_app: FastAPI):
    try:
        if os.getenv("ENV") == "Testing" or os.getenv("ENV") is None:
            manager.load_model(data = data_sample_test.get_content(), file_path = data_sample_test.get_file_path())
        else:
            manager.load_model(data = processor.get_content(), file_path = processor.get_file_path())
        yield
        manager.end_experiment()
    except Exception as e:
        print(f"Error during startup model loading: {e}")

app = FastAPI(
    title="Transformer Model API",
    description="Transformer Model API",
    lifespan=lifespan,
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc"
)

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.post("/process_file/")
async def process_file(file: UploadFile = File(...)):
    global processor

    temp_file_path = Path(f"temp_files/{file.filename}")
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Now process the saved file
    processor = DocumentProcessor(str(temp_file_path))
    processor.process()

    # Get the name and content of the processed text file
    processed_file_path = processor.get_file_path()
    content = processor.get_content()

    return {
        "file_name": processed_file_path.name,
        "content": content
    }

@app.post("/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate_text(request: GenerationRequest):
    try:
        generated = manager.generate_text(request.prompt, request.max_new_tokens)
        return GenerationResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)