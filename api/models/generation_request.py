from pydantic import BaseModel

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50