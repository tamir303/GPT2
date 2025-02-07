from fastapi import APIRouter, HTTPException

from api import GenerationResponse, GenerationRequest
from src import ModelManager

manager = ModelManager()
router = APIRouter(prefix="/generate", tags=["Generation"])


@router.post("", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        generated = manager.generate_text(request.prompt, request.max_new_tokens)
        return GenerationResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
