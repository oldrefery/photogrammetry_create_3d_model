# Dockerfile
# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Для Apple Silicon
#FROM --platform=linux/amd64 ubuntu:22.04

# Set non-interactive mode for apt to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_QPA_PLATFORM=offscreen

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 libxrender1 libxext6 libxft2 \
    libgl1-mesa-glx libqt5gui5 qt5-qmake cmake git g++ wget build-essential libboost-all-dev \
    libeigen3-dev libsuitesparse-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
    libgflags-dev libatlas-base-dev libsqlite3-dev libglew-dev ca-certificates python3-pip \
    libcgal-dev libceres-dev libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# Install VCG library (required for OpenMVS)
RUN git clone https://github.com/cdcseacave/VCG.git /app/VCG && \
    mkdir -p /usr/local/include/vcglib && \
    cp -r /app/VCG/* /usr/local/include/vcglib && \
    rm -rf /app/VCG

# Set working directory
WORKDIR /app

# Clone and build COLMAP
RUN git clone --recursive https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout 3.7 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF && \
    make -j$(nproc) && \
    make install && \
    rm -rf /app/colmap

# Clone and build OpenMVS
RUN git clone https://github.com/cdcseacave/openMVS.git && \
    mkdir openMVS_build && cd openMVS_build && \
    cmake ../openMVS -DCMAKE_BUILD_TYPE=Release \
                     -DVCG_DIR=/usr/local/include/vcglib \
                     -DCMAKE_PREFIX_PATH=/usr/include/opencv4 && \
    make -j2 && \
    make install && \
    rm -rf /app/openMVS /app/openMVS_build

# Copy Python dependencies
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application source code
COPY . /app

# Set permissions for application files
RUN chmod -R 755 /app

# Specify the command to run the application
CMD ["python3", "/app/src/create_3d_model.py", "--clean"]
