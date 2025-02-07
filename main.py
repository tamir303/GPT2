import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from data import data_sample_test

from api.routes import generate_router, manager
from api.routes import files_router, processor

load_dotenv()

@asynccontextmanager
async def lifespan(f_app: FastAPI):
    try:
        if os.getenv("ENV") == "Testing" or os.getenv("ENV") is None:
            manager.load_model(data=data_sample_test.get_content(), file_path=data_sample_test.get_file_path())
        else:
            if processor.input_path is not None:
                manager.load_model(data=processor.get_content(), file_path=processor.get_file_path())
            else:
                print("File isn't set, using temporary ClimateChange file")
                manager.load_model(data=data_sample_test.get_content(), file_path=data_sample_test.get_file_path())
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

# Include routers
app.include_router(generate_router)
app.include_router(files_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
