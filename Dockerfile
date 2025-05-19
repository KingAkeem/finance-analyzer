# Base image with ROCm and Python
FROM rocm/pytorch:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy your analyzer code
COPY . /app
WORKDIR /app

# Environment variables for AMD GPU
ENV OLLAMA_NO_CUDA=1
ENV OLLAMA_HIP=1
ENV PYTHONUNBUFFERED=1
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0 

# Entrypoint
CMD ["python3", "main.py"]