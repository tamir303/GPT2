# Use an official NVIDIA CUDA image as the base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install Python, pip, and other required system packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure "python" points to Python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade --trusted-host "pypi.org" --trusted-host "files.pythonhosted.org" pip && \
    pip install --trusted-host "pypi.org" --trusted-host "files.pythonhosted.org" -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose port 8000 (or whichever port your app uses)
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
