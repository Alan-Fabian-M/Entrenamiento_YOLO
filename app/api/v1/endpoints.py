"""API endpoints for floor plan ingestion, image preprocessing, and scene compilation."""

import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, status
from fastapi.responses import Response

from app.core.config import settings
from app.models.scene_graph import SceneGraphResponse
from app.services.preprocessor import ImagePreprocessor, PreprocessingError
from app.services.scene_compiler import SceneCompiler

router = APIRouter()


@router.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint providing runtime and dependency information."""
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "opencv_version": cv2.__version__,
        "features": [
            "image_ingestion",
            "shadow_removal_background_division",
            "clahe_contrast_enhancement",
            "adaptive_binarization",
            "pipeline_stages_base64",
            "scene_compiler_yolo_opencv_ocr",
        ],
    }


async def _validate_and_read_image(file: UploadFile) -> bytes:
    """Validate uploaded file type, size, and read bytes."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )

    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {list(settings.ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(contents) > settings.MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)} MB.",
        )

    return contents


@router.post(
    "/preprocess",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Returns the preprocessed image in binary PNG format.",
        },
        400: {"description": "Invalid file format or corrupted image."},
    },
    tags=["Preprocessing"],
)
async def preprocess_floorplan(
    file: UploadFile = File(..., description="Floor plan image file (JPG, PNG, WEBP)"),
    filter_mode: str = Query(
        "binary",
        description="Filter mode: 'binary' (full cleaning & adaptive threshold), "
        "'shadow_free' (normalized grayscale with CLAHE), "
        "'grayscale' (basic grayscale), 'otsu' (Otsu threshold).",
        pattern="^(binary|shadow_free|grayscale|otsu)$",
    ),
):
    """Receive an uploaded floor plan photograph or scan, apply lighting normalization,

    shadow removal, contrast enhancement, and return the cleaned PNG image.
    """
    contents = await _validate_and_read_image(file)

    try:
        output_bytes, media_type = ImagePreprocessor.process(contents, filter_mode=filter_mode)
    except PreprocessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return Response(
        content=output_bytes,
        media_type=media_type,
        headers={
            "X-Preprocess-Mode": filter_mode,
            "X-Source-Filename": file.filename or "unknown",
        },
    )


@router.post(
    "/preprocess/pipeline",
    tags=["Preprocessing"],
)
async def preprocess_pipeline_details(
    file: UploadFile = File(..., description="Floor plan image file (JPG, PNG, WEBP)"),
):
    """Receive a floor plan image and return detailed timing metrics alongside

    Base64 encodings of every intermediate stage:
    1. Grayscale
    2. Shadow-free background division
    3. CLAHE local contrast enhancement
    4. Adaptive binarization & noise cleaning

    Allows subsequent AI / Vision sensors (YOLO furniture detector, OpenCV contours, OCR)
    to inspect and pick the optimal representation for their task.
    """
    contents = await _validate_and_read_image(file)

    try:
        pipeline_data = ImagePreprocessor.process_pipeline_detailed(contents)
    except PreprocessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return {
        "status": "success",
        "filename": file.filename,
        "pipeline": pipeline_data,
    }


@router.post(
    "/compilar-sala",
    response_model=SceneGraphResponse,
    tags=["Scene Compiler"],
    summary="Compile a floor plan sketch photo into a Scene Graph JSON for Unity",
)
async def compilar_sala(
    file: UploadFile = File(None, description="Floor plan / croquis photo (JPG, PNG, WEBP)"),
    archivo: UploadFile = File(None, description="Alias for Unity CroquisSceneCompilerController (archivo)"),
):
    upload_file = file or archivo
    if upload_file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Use field 'file' or 'archivo'.",
        )

    contents = await _validate_and_read_image(upload_file)

    try:
        return SceneCompiler.compile(contents)
    except PreprocessingError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
