from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

from backend.api import electives, partner_mock, v1
from backend.api.routes import router
from nlp.component import COMPONENT_VERSION, ModelsUnavailable

UI = Path(__file__).parent / "static"

app = FastAPI(title="NMIMS University Administrative Copilot - NLP API", version=COMPONENT_VERSION,
              description="Elective advisor (syllabi + academic identity -> the elective to enrol in), intent "
                          "classification, routing, entity extraction, retrieval over official NMIMS sources and "
                          "structured service requests, with confidence scores. Demo UI at /ui/. "
                          "See docs/elective_advisor.md and docs/nlp_component.md.")
app.include_router(router)
app.include_router(v1.router)
app.include_router(electives.router)
app.include_router(partner_mock.router)
app.mount("/ui", StaticFiles(directory=UI, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")


@app.exception_handler(ModelsUnavailable)
async def models_unavailable(request: Request, exc: ModelsUnavailable):
    return JSONResponse(status_code=503, content={"error": "models_unavailable", "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": detail})


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000)
