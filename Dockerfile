# Use an official lightweight Python image as a base.
# This keeps our final container size smaller and more secure.
FROM python:3.9-slim

# Set the working directory inside the container to /app.
# All subsequent commands will be run from this directory.
WORKDIR /app

# Install system dependencies. This is a critical step.
# We need 'git' and 'git-lfs' inside the container so it can download
# your large model files from the Git LFS server during the build process on Render.
RUN apt-get update && apt-get install -y git git-lfs && git-lfs install

# Copy your entire local project (including the LFS pointers) into the container.
COPY . .

# --- THIS IS THE CRITICAL FIX ---
# Use Git LFS to pull the actual large model files. This command reads the
# pointer files in the models/ directory and downloads the full .keras and .pkl files.
RUN git lfs pull

# Install all the Python dependencies listed in your requirements file.
# The --no-cache-dir flag is a best practice for keeping container images small.
RUN pip install --no-cache-dir -r requirements.txt

# The command to run your web application using a production-grade server (Gunicorn).
# Render will automatically provide the correct $PORT variable for the server to listen on.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "4", "app:app"]