FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV AEGIT_BRANCH=main
ENV NOTEBOOK_PATH=notebooks/training_unet.ipynb

# Install Python tools
RUN pip install --upgrade pip setuptools wheel toml \
    && pip install jupyter papermill

# Clone and install Aegear
CMD ["bash", "-c", "\
    echo 'Cloning Aegear branch: $AEGIT_BRANCH' && \
    git clone --branch $AEGIT_BRANCH --depth 1 https://github.com/ljubobratovicrelja/aegear.git /aegear && \
    cd /aegear && \
    echo 'Extracting dependencies (excluding torch*)...' && \
    python3 -c \"import toml; d=toml.load('pyproject.toml'); \
    deps = d['project']['dependencies'] + d['project']['optional-dependencies']['dev']; \
    deps = [x for x in deps if not x.startswith('torch')]; \
    print('\\n'.join(deps))\" > clean_reqs.txt && \
    pip install -r clean_reqs.txt && \
    pip install . --no-deps && \
    echo 'Running notebook: $NOTEBOOK_PATH' && \
    mkdir -p /aegear/output && \
    papermill \"$NOTEBOOK_PATH\" \"/aegear/output/executed.ipynb\""]