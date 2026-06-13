# PrivaMed

**Anonimización inteligente de radiografías médicas**

PrivaMed es una solución end-to-end para la desidentificación automática de radiografías médicas. Detecta y redacta información sensible del paciente (nombre, ID, edad, fecha, hora) utilizando un modelo YOLOv8 entrenado, cumpliendo con los requisitos de privacidad del RGPD/LOPD.

## Arquitectura

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   API REST   │────▶│  Modelo YOLO │
│  (Streamlit) │◀────│  (FastAPI)   │◀────│   (YOLOv8n)  │
│  :8501       │     │  :8010       │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

| Servicio | Puerto | Tecnología |
|----------|--------|------------|
| API | 8010 | FastAPI + Uvicorn |
| Frontend | 8501 | Streamlit |
| Train | — | Python + Ultralytics |

## Requisitos

- Docker
- Docker Compose

## Inicio rápido

```bash
# 1. Entrenar el modelo
docker compose --profile train run --rm train

# 2. Levantar la aplicación
docker compose up --build
```

- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8010/docs
- **ReDoc:** http://localhost:8010/redoc

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio |
| `GET` | `/model/info` | Métricas del modelo |
| `GET` | `/classes` | Clases detectables |
| `POST` | `/detect` | Detectar regiones de texto (JSON) |
| `POST` | `/preview` | Preview con bounding boxes |
| `POST` | `/anonymize` | Anonimizar una imagen (PNG) |
| `POST` | `/anonymize/batch` | Anonimizar lote (ZIP) |

## Métricas del modelo

| Métrica | Valor |
|---------|-------|
| mAP@50 | 98.71% |
| mAP@50-95 | 90.73% |
| Precisión | 99.38% |
| Recall | 98.29% |

## Clases detectadas

| ID | Clase | Redacción por defecto |
|----|-------|----------------------|
| 0 | name | Sí |
| 1 | id | Sí |
| 2 | age | Sí |
| 3 | date | No |
| 4 | time | No |

## Dataset

El modelo se entrena con un dataset de ~400 radiografías médicas anotadas en formato YOLO:

- **400 imágenes** en formato PNG (320 train / 80 val)
- **5 clases** de texto sensible: name, id, age, date, time
- Anotaciones con bounding boxes normalizados
- Mezcla de imágenes reales y datos sintéticos

## Estructura del proyecto

```
├── api/                  # Backend FastAPI
│   ├── main.py           # Endpoints REST
│   ├── anonymizer.py     # Lógica de detección y redacción
│   └── trainer.py        # Gestión de metadatos del modelo
├── app/                  # Frontend Streamlit
│   └── main.py           # Dashboard principal
├── images/               # Dataset de radiografías
│   ├── train/            # ~320 imágenes
│   └── val/              # ~80 imágenes
├── labels/               # Anotaciones YOLO
│   ├── train/
│   └── val/
├── models/               # Modelo entrenado (generado)
├── train.py              # Script de entrenamiento
├── data.yaml             # Configuración del dataset
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Stack tecnológico

- **Python 3.11** — lenguaje principal
- **YOLOv8 (Ultralytics)** — detección de objetos
- **FastAPI** — API REST con documentación interactiva
- **Streamlit** — dashboard visual
- **OpenCV** — procesamiento de imagen
- **Docker Compose** — orquestación de servicios

## Hackathon

Proyecto desarrollado para el **Hackathon de Computer Vision** organizado por **IABiomed — Universidad de León** (Junio 2026).

Reto: Computer Vision + OCR · Desidentificación automática · Pipeline end-to-end práctico.
