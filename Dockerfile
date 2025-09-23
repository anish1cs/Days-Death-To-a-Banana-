FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git git-lfs build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# (❌ Removed RUN git lfs pull)

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "app:app", "--workers=4", "--threads=2", "--timeout=120", "--bind", "0.0.0.0:5000"]
