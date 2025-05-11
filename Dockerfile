FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

LABEL maintainer="Relja Ljubobratovic <ljubobratovic.relja@gmail.com>"
LABEL description="Docker image for Aegear with CUDA, branch switching, and notebook execution"

WORKDIR /app

# Install OS and Python tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel toml

# Copy pyproject.toml and install deps excluding torch*
COPY pyproject.toml .

RUN python3 -c "import toml; d=toml.load('pyproject.toml'); \
    deps = d['project']['dependencies']; \
    deps = [x for x in deps if not x.startswith('torch')]; \
    print('\n'.join(deps))" > clean_reqs.txt && \
    pip install -r clean_reqs.txt

RUN python3 -c "import toml; d=toml.load('pyproject.toml'); \
    dev = d['project']['optional-dependencies']['dev']; \
    dev = [x for x in dev if not x.startswith('torch')]; \
    print('\n'.join(dev))" > dev_reqs.txt && \
    pip install -r dev_reqs.txt

# Copy full project and install (no deps to avoid torch override)
COPY . .
RUN pip install . --no-deps

# Install Jupyter and Papermill for notebook execution
RUN pip install jupyter papermill

# Environment variables for runtime control
ENV AEGIT_BRANCH=main
ENV NOTEBOOK_PATH=notebooks/train.ipynb

# Expose Jupyter port (optional, for development)
EXPOSE 8888

# Entry script that switches branch and runs notebook
CMD bash -c "\
    echo 'Checking out branch: $AEGIT_BRANCH' && \
    git fetch origin $AEGIT_BRANCH && \
    git checkout $AEGIT_BRANCH && \
    echo 'Running notebook: $NOTEBOOK_PATH' && \
    mkdir -p /app/output && \
    papermill \"$NOTEBOOK_PATH\" \"/app/output/executed.ipynb\""
