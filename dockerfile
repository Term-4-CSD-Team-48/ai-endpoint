# ----------- Builder Stage ----------- #
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04 AS builder

# Install required packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \ 
    curl \
    git \
    g++ \ 
    libbz2-dev \
    libffi-dev \
    libpcre3-dev \
    libssl-dev \
    libx264-dev \ 
    pkg-config \
    wget \
    yasm \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and extract nginx as well as its rtmp-module into /tmp
WORKDIR /tmp
ENV NGINX_VERSION=1.24.0
RUN curl -O http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz && \
    tar -zxvf nginx-${NGINX_VERSION}.tar.gz
RUN git clone https://github.com/arut/nginx-rtmp-module.git
WORKDIR /tmp/nginx-${NGINX_VERSION}
RUN ./configure --prefix=/opt/nginx \
    --with-http_ssl_module \
    --with-http_auth_request_module \
    --with-http_realip_module \
    --add-module=../nginx-rtmp-module && \
    make -j$(nproc) && make install

# Download and install Python 3.10.0 and pip and add to PATH
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations --prefix=/opt/python3.10 && \
    make -j $(nproc) && \
    make install && \
    rm -rf /tmp/Python-3.10.0*
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    /opt/python3.10/bin/python3 get-pip.py && \
    rm get-pip.py
ENV PATH="/opt/python3.10/bin:$PATH"

# Download and build ffmpeg
WORKDIR /tmp
RUN curl -L https://ffmpeg.org/releases/ffmpeg-6.0.tar.gz | tar xz
WORKDIR /tmp/ffmpeg-6.0
RUN ./configure \
    --prefix=/usr/local \
    --disable-shared \
    --enable-static \
    --enable-libx264 \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --enable-gpl \
    --enable-postproc \
    && make -j$(nproc) && make install

# Copy everything into /app
WORKDIR /app
COPY . . 

# Build python project and install pip dependencies
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
RUN python3 -m pip install -e .
RUN python3 -m pip install -r requirements.txt

# Download .pt files from ./checkpoints
WORKDIR /app/checkpoints
RUN ./download_ckpts.sh

# ----------- Final Stage ----------- #
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

# Copy Python, nginx, and FFmpeg  from builder stage
COPY --from=builder /opt/python3.10 /opt/python3.10
COPY --from=builder /opt/nginx /opt/nginx
COPY --from=builder /usr/lib/x86_64-linux-gnu/libav* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg

# FFmpeg dependencies
RUN apt-get update && apt-get install -y libx264-dev && rm -rf /var/lib/apt/lists/*

# Set up Python symlinks
RUN ln -sf /opt/python3.10/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /opt/python3.10/bin/python3.10 /usr/local/bin/python && \
    ln -sf /opt/python3.10/bin/pip3.10 /usr/local/bin/pip3 && \
    ln -sf /opt/python3.10/bin/pip3.10 /usr/local/bin/pip && \
    python --version && \
    python3 --version && \
    pip --version && \
    pip3 --version 

# Python runs in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Set up nginx symlink
ENV PATH="/opt/nginx/sbin:${PATH}"
RUN nginx -v

# Set up nginx logs
RUN mkdir -p /var/log/nginx

# Copy built project into /app
WORKDIR /app
COPY --from=builder /app ./

# Expose port for nginx
EXPOSE 8080

# Expose port for RTMP
EXPOSE 1935

# Make executable serve.sh
RUN chmod +x ./serve.sh

# Set the entrypoint
ENTRYPOINT ["./serve.sh"]
