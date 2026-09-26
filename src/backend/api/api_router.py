from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from logic.intent.predict import classify
from logic.pipeline.copilot import get_copilot

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


class CopilotRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, examples=["I lost my ID card. What should I do?"])


@router.post('/copilot/query')
def copilot_query(request: CopilotRequest):
    try:
        return get_copilot().run(request.query)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"{e}. Run: python -m logic.pipeline.train_v1")
