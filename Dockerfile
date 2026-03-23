FROM python:3.13-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY ml/ ml/
COPY data/ data/
COPY scripts/ scripts/

# Pre-download YOLO weights (cached in image layer)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" || true

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
