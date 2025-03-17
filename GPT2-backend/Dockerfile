FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install system dependencies and Python 3.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the FastAPI server (using uvicorn)
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
