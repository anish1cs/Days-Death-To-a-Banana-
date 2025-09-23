# Use an official lightweight Python image as a base.
FROM python:3.9-slim

# Set the working directory inside the container to /app.
WORKDIR /app

# Install system dependencies: git and git-lfs are crucial.
RUN apt-get update && apt-get install -y git git-lfs && git-lfs install

# --- THIS IS THE CRITICAL FIX ---
# Instead of copying local files, we clone the repository directly from GitHub.
# This ensures that the .git directory is present inside the container,
# which is necessary for `git lfs pull` to work.
# This uses your actual public repository URL.
RUN git clone https://github.com/anish1cs/Days-Death-To-a-Banana-.git .

# Now that we are in a proper Git repository, this command will succeed.
# It reads the pointer files and downloads the actual large model files.
RUN git lfs pull

# Install all the Python dependencies from your requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# The command to run your web application using a production-grade server (Gunicorn).
# Render will automatically provide the correct $PORT variable.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "4", "app:app"]