FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY app/ ./app/
COPY train.py .
COPY data.yaml .
COPY images/ ./images/
COPY labels/ ./labels/

RUN find /app/labels -name "*.cache" -delete 2>/dev/null; \
    echo "=== Images train ===" && ls /app/images/train/ | head -3 && \
    echo "=== Labels train ===" && ls /app/labels/train/ | head -3 && \
    echo "=== Image count ===" && ls /app/images/train/ | wc -l && \
    echo "=== Label count ===" && ls /app/labels/train/ | wc -l

EXPOSE 8000 8501
