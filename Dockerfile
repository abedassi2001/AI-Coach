# AI Gym Form Coach — development image
# Build: docker build -t ai-gym-form-coach .
# Run:   docker run -p 8000:8000 -v "%cd%":/app ai-gym-form-coach

FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: API server (Phase 10+)
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-c", "print('AI Gym Form Coach container ready. Override CMD for your task.')"]
