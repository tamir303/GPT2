from pydantic import BaseModel


class InferenceRequest(BaseModel):
    context: str

class InferenceResponse(BaseModel):
    generated_text: str