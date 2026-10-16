from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import uvicorn

from backend.api import v1
from backend.api.routes import router
from nlp.component import COMPONENT_VERSION, ModelsUnavailable

app = FastAPI(title="NMIMS University Administrative Copilot - NLP API", version=COMPONENT_VERSION,
              description="Intent classification, sub-intent routing, entity extraction, retrieval over official "
                          "NMIMS sources and structured service requests, with confidence scores. "
                          "See docs/nlp_component.md.")
app.include_router(router)
app.include_router(v1.router)


@app.exception_handler(ModelsUnavailable)
async def models_unavailable(request: Request, exc: ModelsUnavailable):
    return JSONResponse(status_code=503, content={"error": "models_unavailable", "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": detail})


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000)
