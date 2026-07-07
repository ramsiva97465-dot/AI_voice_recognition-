# ---------------------------------------------------------------
# Multi-stage Dockerfile for Voice Authentication API
# Base: python:3.11-slim + system audio/build deps
# ---------------------------------------------------------------

FROM python:3.11-slim

# ----- System dependencies -----
# libsndfile1   -> soundfile / librosa
# ffmpeg        -> torchaudio audio I/O
# libgomp1      -> PyTorch OpenMP
# gcc / g++     -> build wheels for psycopg2, pgvector, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ----- Working directory -----
WORKDIR /app

# ----- Install Python dependencies -----
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch and Torchaudio first to save 2GB+ space and speed up Railway builds
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# ----- Copy project files -----
COPY . .

# ----- Runtime -----
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Trigger Railway Rebuild: 2 (CPU-only torch + all latest changes)

