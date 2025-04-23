# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04 as builder

# Install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \ 
    libbz2-dev \
    libffi-dev \
    libssl-dev \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir to /tmp
WORKDIR /tmp

# Download and install Python 3.10.0
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations --prefix=/opt/python3.10 && \
    make -j $(nproc) && \
    make install && \
    rm -rf /tmp/Python-3.10.0*

# Set workdir to /app
WORKDIR /app

# Copy everything
COPY . . 

# Download .pt files from ./checkpoints
RUN ./checkpoints/download_ckpts.sh

# Build project
RUN python3 -m pip install torch torchvision torchaudio
RUN python3 -m pip install -e .

# Final stage - smaller runtime image
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

# Copy Python from builder stage
COPY --from=builder /opt/python3.10 /opt/python3.10

# Set workdir to /app
WORKDIR /app

# Set up Python symlinks
RUN ln -sf /opt/python3.10/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /opt/python3.10/bin/python3.10 /usr/local/bin/python && \
    ln -sf /opt/python3.10/bin/pip3.10 /usr/local/bin/pip3 && \
    ln -sf /opt/python3.10/bin/pip3.10 /usr/local/bin/pip && \
    python --version && \
    python3 --version && \
    pip --version && \
    pip3 --version 

# Copy built project
COPY --from=builder /app ./app

# Install nginx
RUN apt-get update && apt-get install -y \
    nginx \
    && rm -rf /var/lib/apt/lists/* && nginx -v

# Expose port for nginx
EXPOSE 8080

# Expose port for RTMP
EXPOSE 1935

# Make executable serve.sh
RUN chmod +x ./serve.sh

# Set the entrypoint
ENTRYPOINT ["./serve.sh"]
