# Use an official Python 3.10.6 base image
FROM python:3.10.6

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
ENV PYTHONUNBUFFERED=TRUE

# Install dependencies and Python 3.10.6
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        git \ 
        wget \
        nginx \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install pytorch 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install dependencies (requirements.txt)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install dependencies (setup.py)
COPY setup.py .
COPY .clang-format .
COPY README.md .
COPY pyproject.toml .
COPY sam2 .
RUN pip install -e .

COPY checkpoints .
COPY predictor.py .
COPY serve.py .
COPY nginx.conf .
ENTRYPOINT ["python", "predictor.py"]