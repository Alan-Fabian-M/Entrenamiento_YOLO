"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    Microservicio local de visión artificial para digitalización y preprocesamiento
    de planos arquitectónicos 2D.
    
    Parte del pipeline de InmobiliariaVR (ISW2):
    1. Ingestión y preprocesamiento (Filtros, sombras, CLAHE, binarización).
    2. Detección de mobiliario (YOLO OBB) y paredes (OpenCV).
    3. Extracción de cotas (OCR) y motor métrico (píxeles a metros).
    4. Ensamblaje determinista 3D en Unity (Meta Quest / Android AR).
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Unity, Web clients, and local tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers (both versioned /api/v1 and root / for legacy Unity controllers)
app.include_router(api_v1_router, prefix=settings.API_V1_STR)
app.include_router(api_v1_router, prefix="")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
