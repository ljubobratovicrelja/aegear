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

# Set default environment variables
ENV AEGEAR_BRANCH=main
ENV NOTEBOOK_PATH=notebooks/training_unet.ipynb

# Install Python tools
RUN pip install --upgrade pip setuptools wheel toml \
    && pip install jupyter papermill

# Runtime: clone, install, run notebook
CMD ["bash", "-c", "\
    echo \"Preparing Aegear repo in /app\" && \
    if [ -d /app/.git ]; then \
    echo 'Repo already exists, resetting to branch $AEGEAR_BRANCH...' && \
    cd /app && \
    git fetch origin $AEGEAR_BRANCH && \
    git reset --hard origin/$AEGEAR_BRANCH; \
    else \
    echo 'Cloning fresh repo...' && \
    git clone --branch $AEGEAR_BRANCH --depth 1 https://github.com/ljubobratovicrelja/aegear.git /app; \
    fi && \
    cd /app && \
    echo 'Latest commit:' && git log -1 --pretty=format:'Commit: %h - %s' && \
    echo 'Extracting dependencies (excluding torch*)...' && \
    python3 -c \"import toml; d=toml.load('pyproject.toml'); \
    deps = d['project']['dependencies'] + d['project']['optional-dependencies']['dev']; \
    deps = [x for x in deps if not x.startswith('torch')]; \
    print('\\\\n'.join(deps))\" > clean_reqs.txt && \
    pip install -r clean_reqs.txt && \
    pip install . --no-deps && \
    echo \"Running notebook: $NOTEBOOK_PATH\" && \
    mkdir -p /app/output && \
    cd /app/notebooks && \
    papermill $(basename $NOTEBOOK_PATH) /app/output/executed.ipynb"]

