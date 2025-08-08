FROM ubuntu:22.04

# Set environment variables
ENV ERT_SHOW_BACKTRACE=1
ENV UV_SYSTEM_PYTHON=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    doxygen \
    libegl1 \
    liblapack-dev \
    libblas-dev \
    gfortran \
    software-properties-common \
    wget \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | bash

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install build dependencies
RUN python3.11 -m pip install meson ninja

# Install package with test dependencies
RUN uv pip install ".[test]"

# Run tests
CMD ["pytest", "tests/"]
