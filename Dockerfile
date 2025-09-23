# -------------------------
# 1. Base Image
# -------------------------
FROM python:3.9-slim

# -------------------------
# 2. System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    git git-lfs build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 3. Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# 4. Copy project files
# -------------------------
COPY . .

# -------------------------
# 5. Ensure Git LFS models are pulled
# -------------------------
RUN git lfs pull

# -------------------------
# 6. Install Python dependencies
# -------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------
# 7. Expose port
# -------------------------
EXPOSE 5000

# -------------------------
# 8. Run app with Gunicorn
# -------------------------
CMD ["gunicorn", "app:app", "--workers=4", "--threads=2", "--timeout=120", "--bind", "0.0.0.0:5000"]
