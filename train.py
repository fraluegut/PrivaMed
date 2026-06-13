import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = "data.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 16
MODEL_BASE = "yolov8n.pt"
PROJECT = "models"
NAME = "radshield"
META_FILE = Path(PROJECT) / NAME / "train_meta.json"


def main():
    for cache in Path("labels").rglob("*.cache"):
        cache.unlink(missing_ok=True)
        print(f"Deleted stale cache: {cache}")

    print(f"\n{'='*60}")
    print(f"  PrivaMed — Entrenamiento del modelo")
    print(f"  Épocas: {EPOCHS} | Imagen: {IMGSZ}px | Batch: {BATCH}")
    print(f"{'='*60}\n")

    model = YOLO(MODEL_BASE)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
        workers=0,
    )

    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    last = Path(PROJECT) / NAME / "weights" / "last.pt"
    weights = best if best.exists() else last

    print(f"\nPesos guardados en: {weights}")
    print("Ejecutando validación final...")

    val_model = YOLO(str(weights))
    val_results = val_model.val(data=DATA_YAML, verbose=True)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "base_model": MODEL_BASE,
        "metrics": {
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50-95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4),
        },
        "best_weights": str(weights),
    }

    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  Entrenamiento completado")
    print(f"  mAP@50:    {meta['metrics']['mAP50']:.4f}")
    print(f"  mAP@50-95: {meta['metrics']['mAP50-95']:.4f}")
    print(f"  Precisión: {meta['metrics']['precision']:.4f}")
    print(f"  Recall:    {meta['metrics']['recall']:.4f}")
    print(f"  Metadata:  {META_FILE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
