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

# Copy your local project files into the container at the /app directory.
# This copies app.py, requirements.txt, models/, templates/, etc.
COPY . .

# Use Git LFS to pull the actual large model files.
# This command reads the pointer files (that were copied above) and downloads
# the full .keras and .pkl files into the /app/models/ directory inside the container.
RUN git lfs pull

# Install all the Python dependencies listed in your requirements file.
# The --no-cache-dir flag is a best practice for keeping container images small.
RUN pip install --no-cache-dir -r requirements.txt

# The command to run your web application using a production-grade server (Gunicorn).
# It will run the 'app' object inside your 'app.py' file.
# Render will automatically provide the correct $PORT variable for the server to listen on.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "4", "app:app"]

