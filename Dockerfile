# Use a slim Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files & force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (for TensorFlow, Pillow, LightGBM, OpenCV)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies (CPU-only TensorFlow)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask/Gunicorn port
EXPOSE 5000

# Start app with Gunicorn
# Lower workers to reduce memory usage (Render free tier ~512MB RAM)
CMD ["gunicorn", "--workers=2", "--threads=2", "--timeout=120", "--bind=0.0.0.0:5000", "app:app"]
