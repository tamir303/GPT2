from typing import List

from pydantic import BaseModel

class TokenizerEncodeRequest(BaseModel):
    raw: str

class TokenizerDecodeRequest(BaseModel):
    token_ids: List[int]

class TokenizerEncodeResponse(BaseModel):
    encoded_text: str
    encoded_ids: List[int]

class TokenizerDecodeResponse(BaseModel):
    decoded_text: str
    decoded_tokens: List[int]