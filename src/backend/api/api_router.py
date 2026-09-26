from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from logic.intent.predict import classify

router = APIRouter()


class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["I lost my ID card. What should I do?"])


@router.get('/health')
async def health():
    return {
        "status": "ok"
    }


@router.post('/intent/classify')
async def classify_intent(request: IntentRequest):
    try:
        return classify(request.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
