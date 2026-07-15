# ---------------------------------------------------------------
# Dockerfile for Voice Authentication API - Railway Optimized
# ---------------------------------------------------------------

FROM python:3.11-slim

# ----- System dependencies -----
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
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Step 1: Install CPU-only PyTorch first (pinned versions = faster pip resolve)
# Correct syntax: specify --index-url AFTER package names
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
    torchaudio==2.2.2

# Step 2: Install remaining packages (torch already satisfied, won't re-download)
RUN pip install --no-cache-dir -r requirements.txt

# ----- Copy project files -----
COPY . .

# ----- Runtime -----
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

