from pydantic import BaseModel

class InferenceRequest(BaseModel):
    prompt: str