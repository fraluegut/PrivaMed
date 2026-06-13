import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


META_FILE = "models/radshield/train_meta.json"


def _clear_label_cache():
    for cache in Path("/app/labels").rglob("*.cache"):
        cache.unlink(missing_ok=True)


def train_model(
    data_yaml: str = "data.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    model_base: str = "yolov8n.pt",
    project: str = "models",
    name: str = "radshield",
    progress_callback=None,
) -> tuple[Path, dict]:
    _clear_label_cache()

    model = YOLO(model_base)

    def on_train_epoch_end(trainer):
        if progress_callback is None:
            return
        current = trainer.epoch + 1
        total = trainer.epochs
        metrics = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            metrics = trainer.metrics
        progress_callback(current, total, metrics)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    best = Path(project) / name / "weights" / "best.pt"
    last = Path(project) / name / "weights" / "last.pt"
    weights = best if best.exists() else last

    val_metrics = {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}
    try:
        val_model = YOLO(str(weights))
        val_results = val_model.val(data=data_yaml, verbose=False)
        val_metrics = {
            "mAP50": round(float(val_results.box.map50), 4),
            "mAP50-95": round(float(val_results.box.map), 4),
            "precision": round(float(val_results.box.mp), 4),
            "recall": round(float(val_results.box.mr), 4),
        }
    except Exception:
        pass

    meta = {
        "trained_at": datetime.now().isoformat(),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "base_model": model_base,
        "metrics": val_metrics,
        "best_weights": str(weights),
    }
    meta_path = Path(META_FILE)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return weights, meta


def load_train_meta() -> dict | None:
    meta_path = Path(META_FILE)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())
