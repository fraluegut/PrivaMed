from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = {0: "name", 1: "id", 2: "age", 3: "date", 4: "time"}
SENSITIVE_CLASSES = {0, 1, 2}  # name, id, age — always redact
OPTIONAL_CLASSES = {3, 4}      # date, time — user decides


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def detect_text_regions(model: YOLO, image: np.ndarray, conf: float = 0.25):
    results = model.predict(image, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, "unknown"),
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            })
    return detections


def redact_image(
    image: np.ndarray,
    detections: list[dict],
    classes_to_redact: set[int],
    margin: int = 5,
) -> np.ndarray:
    redacted = image.copy()
    h, w = redacted.shape[:2]
    for det in detections:
        if det["class_id"] not in classes_to_redact:
            continue
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        redacted[y1:y2, x1:x2] = 0
    return redacted


def annotate_preview(
    image: np.ndarray,
    detections: list[dict],
    classes_to_redact: set[int],
) -> np.ndarray:
    preview = image.copy()
    if len(preview.shape) == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    elif preview.shape[2] == 1:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        will_redact = det["class_id"] in classes_to_redact
        color = (0, 0, 255) if will_redact else (0, 200, 0)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.putText(preview, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return preview


def process_single(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    classes_to_redact: set[int],
    conf: float = 0.25,
    margin: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    detections = detect_text_regions(model, image, conf=conf)
    preview = annotate_preview(image, detections, classes_to_redact)
    redacted = redact_image(image, detections, classes_to_redact, margin=margin)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_anon{image_path.suffix}"
    cv2.imwrite(str(out_path), redacted)

    return preview, redacted, detections
