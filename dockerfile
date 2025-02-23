FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
ENV PYTHONUNBUFFERED=TRUE

# Install dependencies and Python 3.10.6
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \         
        build-essential \
        cmake \
        ninja-build \
        git \ 
        wget \
        nginx \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link from python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install pytorch 
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install dependencies (setup.py)
COPY setup.py .
COPY .clang-format .
COPY README.md .
COPY pyproject.toml .
COPY sam2 .
COPY checkpoints .
RUN python3 -m pip install -e .

# Install dependencies (requirements.txt)
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt 

COPY predictor.py .
COPY serve .
RUN chmod +x serve
COPY nginx.conf .
COPY wsgi.py .

# Create necessary directories
RUN mkdir -p /nginx /var/log/nginx

# Expose port 8080
EXPOSE 8080

ENTRYPOINT ["./serve"]