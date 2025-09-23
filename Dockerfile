# Use lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files & set buffer flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV, Pillow, LightGBM)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask/Gunicorn port
EXPOSE 5000

# Run the app with Gunicorn (better for production)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
