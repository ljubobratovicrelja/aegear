FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

LABEL maintainer="Relja Ljubobratovic <ljubobratovic.relja@gmail.com>"
LABEL description="Aegear training image with runtime branch selection and notebook execution"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV AEGEAR_BRANCH=main

RUN pip install --upgrade pip setuptools wheel toml

# Copy the run_training.sh script into the image
COPY run_training.sh /app/run_training.sh

# Use the script as the entrypoint
CMD ["bash", "/app/run_training.sh"]

