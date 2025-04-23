# Use NVIDIA CUDA base image with Ubuntu
FROM ubuntu:20.04 as builder

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
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

# Download .pt files from ./checkpoints
COPY checkpoints ./
RUN ./checkpoints/download_ckpts.sh

# Final stage - smaller runtime image
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

# Set workdir to /app
WORKDIR /app

# Copy Python from builder stage
COPY --from=builder /opt/python3.10 /opt/python3.10

# Set up Python symlinks
RUN ln -sf /opt/python3.10/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /opt/python3.10/bin/python3.10 /usr/local/bin/python && \
    ln -sf /opt/python3.10/bin/pip3.10 /usr/local/bin/pip3 && \
    ln -sf /opt/python3.10/bin/pip3.10 /usr/local/bin/pip

# Copy .pt files from builder stage
COPY --from=builder /tmp/checkpoints ./

# Install nginx
RUN apt-get update && apt-get install -y \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN python3 --version && \
    pip3 --version && \
    nginx -v

# Set up dir to copy project files
COPY .clang-format \
    clang-format.txt \
    CODE_OF_CONDUCT.md \
    CONTRIBUTING.md \
    LICENSE \ 
    LICENSE_cctorch \ 
    my_app.py \
    nginx.conf \
    processes.py \
    pyproject.toml \
    README.md \
    sam2 \
    serve \
    serve.sh \
    setup.py \
    streamer.py \
    tracker.py \
    wsgi.py \
    ./

# Expose port for nginx
EXPOSE 8080

# Expose port for RTMP
EXPOSE 1935

# Set the entrypoint
ENTRYPOINT ["/serve.sh"]
