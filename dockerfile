# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install required packages
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    curl \
    libbz2-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.10.0
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python && \
    rm -rf /tmp/Python-3.10.0*

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install PyTorch and torchvision with CUDA support
RUN pip3 install torch>=2.3.1 torchvision>=0.18.1

# Set the entrypoint
ENTRYPOINT ["/serve.sh"]
