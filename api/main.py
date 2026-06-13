import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .anonymizer import (
    CLASS_NAMES,
    SENSITIVE_CLASSES,
    annotate_preview,
    detect_text_regions,
    load_model,
    redact_image,
)
from .trainer import load_train_meta

APP_TITLE = "PrivaMed API"
APP_DESCRIPTION = """
API de anonimización de radiografías médicas.

Detecta y redacta información sensible (nombre, id, edad, fecha, hora)
en imágenes de radiografías utilizando un modelo YOLOv8 entrenado.
"""

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("models/radshield/weights/best.pt")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            return None
        _model = load_model(str(MODEL_PATH))
    return _model


def _parse_classes(classes: list[int] | None) -> set[int]:
    if classes is None:
        return set(SENSITIVE_CLASSES)
    return set(classes)


def _read_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {file.filename}")
    return image


# ── Health & info ────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    """Estado de salud del servicio."""
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/model/info", tags=["Modelo"])
def model_info():
    """Información del modelo entrenado: métricas, fecha, configuración."""
    meta = load_train_meta()
    if meta is None:
        return {"status": "not_trained", "message": "No hay modelo entrenado."}
    return {"status": "trained", **meta}


@app.get("/classes", tags=["Modelo"])
def list_classes():
    """Clases detectables y cuáles se redactan por defecto."""
    return {
        "classes": {
            str(k): {
                "name": v,
                "default_redact": k in SENSITIVE_CLASSES,
            }
            for k, v in CLASS_NAMES.items()
        }
    }


# ── Detection ────────────────────────────────────────────────────────────────

@app.post("/detect", tags=["Procesamiento"])
def detect(
    file: UploadFile = File(..., description="Imagen de radiografía"),
    confidence: float = Query(0.25, ge=0.05, le=0.95, description="Umbral de confianza"),
):
    """Detecta regiones de texto sensible en una radiografía. Devuelve las detecciones sin modificar la imagen."""
    model = get_model()
    if model is None:
        return {"error": "Modelo no disponible. Entrena primero."}

    image = _read_upload(file)
    detections = detect_text_regions(model, image, conf=confidence)

    return {
        "filename": file.filename,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "detections": detections,
        "total": len(detections),
    }


# ── Anonymize single ────────────────────────────────────────────────────────

@app.post("/anonymize", tags=["Procesamiento"])
def anonymize(
    file: UploadFile = File(..., description="Imagen de radiografía"),
    confidence: float = Query(0.25, ge=0.05, le=0.95, description="Umbral de confianza"),
    margin: int = Query(5, ge=0, le=30, description="Margen de redacción en píxeles"),
    classes: list[int] = Query(None, description="IDs de clases a redactar (0=name,1=id,2=age,3=date,4=time). Por defecto: 0,1,2"),
):
    """Anonimiza una radiografía redactando las regiones de texto sensible. Devuelve la imagen anonimizada en PNG."""
    model = get_model()
    if model is None:
        return {"error": "Modelo no disponible. Entrena primero."}

    image = _read_upload(file)
    classes_to_redact = _parse_classes(classes)
    detections = detect_text_regions(model, image, conf=confidence)
    redacted = redact_image(image, detections, classes_to_redact, margin=margin)

    _, encoded = cv2.imencode(".png", redacted)
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{Path(file.filename).stem}_anon.png"'},
    )


# ── Preview (annotated) ─────────────────────────────────────────────────────

@app.post("/preview", tags=["Procesamiento"])
def preview(
    file: UploadFile = File(..., description="Imagen de radiografía"),
    confidence: float = Query(0.25, ge=0.05, le=0.95, description="Umbral de confianza"),
    classes: list[int] = Query(None, description="IDs de clases a redactar"),
):
    """Genera una vista previa con bounding boxes coloreados: rojo=será redactado, verde=se conserva."""
    model = get_model()
    if model is None:
        return {"error": "Modelo no disponible. Entrena primero."}

    image = _read_upload(file)
    classes_to_redact = _parse_classes(classes)
    detections = detect_text_regions(model, image, conf=confidence)
    preview_img = annotate_preview(image, detections, classes_to_redact)

    _, encoded = cv2.imencode(".png", preview_img)
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
    )


# ── Batch anonymize ──────────────────────────────────────────────────────────

@app.post("/anonymize/batch", tags=["Procesamiento"])
def anonymize_batch(
    files: list[UploadFile] = File(..., description="Múltiples imágenes de radiografías"),
    confidence: float = Query(0.25, ge=0.05, le=0.95),
    margin: int = Query(5, ge=0, le=30),
    classes: list[int] = Query(None, description="IDs de clases a redactar"),
):
    """Anonimiza múltiples radiografías. Devuelve un archivo ZIP con todas las imágenes procesadas."""
    model = get_model()
    if model is None:
        return {"error": "Modelo no disponible. Entrena primero."}

    classes_to_redact = _parse_classes(classes)
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            image = _read_upload(f)
            detections = detect_text_regions(model, image, conf=confidence)
            redacted = redact_image(image, detections, classes_to_redact, margin=margin)
            _, encoded = cv2.imencode(".png", redacted)
            zf.writestr(f"{Path(f.filename).stem}_anon.png", encoded.tobytes())

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="radshield_batch.zip"'},
    )
