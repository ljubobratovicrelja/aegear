#!/usr/bin/env bash

# Set TASK_NAME (should be unique per pod/job)
export TASK_NAME=${TASK_NAME:-default_task}
export CLEARML_TASK=$TASK_NAME

# Repo stays in /app (container-local)
REPO_DIR="/app/$TASK_NAME"
# Outputs/checkpoints/logs go to /workspace (persistent volume)
OUTPUT_DIR="/workspace/$TASK_NAME"

if [ -n "$RUNPOD_POD_ID" ]; then
  SINGLE_RUN_SENTINEL="/workspace/.aegear_run_complete_${RUNPOD_POD_ID}"
else
  SINGLE_RUN_SENTINEL="/workspace/.aegear_run_complete"
fi

if [ -n "$RUNPOD_POD_ID" ] && [ -f "$SINGLE_RUN_SENTINEL" ]; then
  echo "============================================"
  echo "TRAINING ALREADY COMPLETED FOR THIS POD"
  echo "============================================"
  echo "Sentinel file exists: $SINGLE_RUN_SENTINEL"
  echo "This indicates training has already run successfully."
  echo "Exiting gracefully to prevent restart loop."
  echo "============================================"
  # Exit with 0 to signal success and prevent RunPod from restarting
  exit 0
fi


# Check if OUTPUT_DIR already exists (to avoid overwriting outputs)
if [ -d "$OUTPUT_DIR" ]; then
  echo "============================================"
  echo "WARNING: Output directory already exists"
  echo "============================================"
  echo "Directory: $OUTPUT_DIR"
  
  # If we're on RunPod and sentinel doesn't exist, this is likely a restart loop
  if [ -n "$RUNPOD_POD_ID" ]; then
    echo "This appears to be a container restart."
    echo "Creating sentinel and exiting to prevent loop."
    touch "$SINGLE_RUN_SENTINEL"
    exit 0
  else
    echo "ERROR: Please use a unique TASK_NAME or remove the existing directory."
    exit 1
  fi
fi


# Clone repo into unique directory (container-local)
echo "Cloning fresh repo into $REPO_DIR..."
git clone --branch "$AEGEAR_BRANCH" --depth 1 https://github.com/ljubobratovicrelja/aegear.git "$REPO_DIR"
cd "$REPO_DIR"

echo 'Latest commit:'
git log -1 --pretty=format:'Commit: %h - %s'

echo  # empty line for better readability

echo 'Installing aegear with training dependencies...'
pip install -q .[train]

if [ -n "$CLEARML_API_ACCESS_KEY" ] && [ -n "$CLEARML_API_SECRET_KEY" ]; then
  echo 'Installing ClearML client for experiment tracking...'
  pip install -q clearml
fi

echo ''
echo '========================================='
echo 'CUDA/GPU DIAGNOSTICS'
echo '========================================='
echo "DEVICE environment variable: $DEVICE"
echo ''
echo 'PyTorch CUDA availability:'
CUDA_AVAILABLE=$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')")
python -c "import torch; print(f'  torch.cuda.is_available(): {torch.cuda.is_available()}'); print(f'  torch.cuda.device_count(): {torch.cuda.device_count()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('  No GPUs detected')"
echo ''
echo 'nvidia-smi output:'
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo '  nvidia-smi not found'
fi
echo '========================================='
echo ''

# Validate CUDA availability if cuda device is requested
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "auto" ]; then
  if [ "$CUDA_AVAILABLE" = "0" ]; then
    echo ''
    echo '========================================='
    echo '❌ CUDA VALIDATION FAILED'
    echo '========================================='
    echo 'CUDA device was requested but torch.cuda.is_available() returned False.'
    echo 'This indicates a machine-level issue (likely driver problems).'
    echo ''
    echo 'Exiting with code 42 to signal retry on a different machine.'
    echo '========================================='
    echo ''
    
    # Mark as complete to prevent restart loops
    if [ -n "$RUNPOD_POD_ID" ]; then
      touch "$SINGLE_RUN_SENTINEL"
      echo "Created sentinel file to prevent restart: $SINGLE_RUN_SENTINEL"
      
      # Self-terminate the pod
      if [ -n "$RUNPOD_API_KEY" ] && command -v curl &> /dev/null; then
        echo "Self-terminating RunPod pod: $RUNPOD_POD_ID"
        curl -s -X POST https://api.runpod.io/graphql \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $RUNPOD_API_KEY" \
          -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" > /dev/null
        echo "✓ Pod termination request sent"
      fi
    fi
    
    exit 42
  else
    echo '✓ CUDA validation passed - GPU is available'
    echo ''
  fi
fi

# Set defaults for all args with predefined values in train.py
export BATCH_SIZE=${BATCH_SIZE:-128}
export TRAIN_RATIO=${TRAIN_RATIO:-0.85}
export NUM_WORKERS=${NUM_WORKERS:-4}
export GAUSSIAN_SIGMA=${GAUSSIAN_SIGMA:-15.0}
export WEIGHTS=${WEIGHTS:-IMAGENET1K_V1}
export PRETRAINED_MODEL_DIR=${PRETRAINED_MODEL_DIR:-$OUTPUT_DIR/models}
export EPOCHS=${EPOCHS:-10}
export LR=${LR:-0.0001}
export EPOCH_VIS=${EPOCH_VIS:-$OUTPUT_DIR/vis_epochs}
export EPOCH_SAVE_INTERVAL=${EPOCH_SAVE_INTERVAL:-1}
export DEVICE=${DEVICE:-auto}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.005}
export ACTIVATION=${ACTIVATION:-relu}
export SCHEDULER_TYPE=${SCHEDULER_TYPE:-}
export SCHEDULER_PARAMS=${SCHEDULER_PARAMS:-}
export SEED=${SEED:-42}

# Set output paths for model and checkpoints
export MODEL_DIR=${MODEL_DIR:-$OUTPUT_DIR/models}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}

# Check required arguments
if [ -z "$MODEL_TYPE" ] || [ -z "$DATA_MANIFEST" ] || [ -z "$MODEL_DIR" ] || [ -z "$CHECKPOINT_DIR" ]; then
  echo 'ERROR: MODEL_TYPE, DATA_MANIFEST, MODEL_DIR, and CHECKPOINT_DIR must be set as environment variables.'
  exit 1
fi

echo ''
echo '========================================='
echo 'TRAINING CONFIGURATION'
echo '========================================='
echo "DEVICE setting: $DEVICE"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "EPOCHS: $EPOCHS"
echo "NUM_WORKERS: $NUM_WORKERS"
echo '========================================='
echo ''


# Write training stages to MODEL_DIR/training_stages.json if set
if [ ! -z "$TRAINING_STAGES" ]; then
  mkdir -p "$MODEL_DIR"
  echo "$TRAINING_STAGES" > "$MODEL_DIR/training_stages.json"
  ARGS="$ARGS --training-stages=$MODEL_DIR/training_stages.json"
fi

ARGS="--model-type=$MODEL_TYPE --data-manifest=$DATA_MANIFEST --batch-size=$BATCH_SIZE --train-ratio=$TRAIN_RATIO --num-workers=$NUM_WORKERS --gaussian-sigma=$GAUSSIAN_SIGMA --weights=$WEIGHTS"
ARGS="$ARGS --model-dir=$MODEL_DIR --pretrained-model-dir=$PRETRAINED_MODEL_DIR --checkpoint-dir=$CHECKPOINT_DIR --epoch-vis=$EPOCH_VIS --epoch-save-interval=$EPOCH_SAVE_INTERVAL --device=$DEVICE --weight-decay=$WEIGHT_DECAY --activation=$ACTIVATION --seed=$SEED"
[ ! -z "$CONTINUE_TRAINING" ] && [ "$CONTINUE_TRAINING" != "0" ] && ARGS="$ARGS --continue-training"
[ ! -z "$USE_BEST_MODEL" ] && [ "$USE_BEST_MODEL" != "0" ] && ARGS="$ARGS --use-best-model"
[ ! -z "$EPOCHS" ] && ARGS="$ARGS --epochs=$EPOCHS"
[ ! -z "$LR" ] && ARGS="$ARGS --lr=$LR"
[ ! -z "$LOSS_PARAMS" ] && ARGS="$ARGS --loss-params=$LOSS_PARAMS"
[ ! -z "$CONFIG" ] && ARGS="$ARGS --config=$CONFIG"
[ ! -z "$CBAM" ] && [ "$CBAM" != "0" ] && ARGS="$ARGS --cbam"
[ ! -z "$CLEARML_TASK" ] && ARGS="$ARGS --clearml-task=$CLEARML_TASK"
[ ! -z "$CLEARML_PROJECT" ] && ARGS="$ARGS --clearml-project=$CLEARML_PROJECT"
[ ! -z "$USE_VISUALIZER" ] && [ "$USE_VISUALIZER" != "0" ] && ARGS="$ARGS --use-visualizer"
[ ! -z "$SCHEDULER_TYPE" ] && ARGS="$ARGS --scheduler-type=$SCHEDULER_TYPE"
[ ! -z "$SCHEDULER_PARAMS" ] && ARGS="$ARGS --scheduler-params=$SCHEDULER_PARAMS"
[ ! -z "$AUTODOWNLOAD" ] && [ "$AUTODOWNLOAD" != "0" ] && ARGS="$ARGS --autodownload"
[ ! -z "$VERBOSE" ] && [ "$VERBOSE" != "0" ] && ARGS="$ARGS --verbose"

echo "Running training script with args: $ARGS"
python tools/train.py $ARGS

# Capture exit code
TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
  echo "✓ Training completed successfully"
else
  echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
fi

# Mark training as complete to prevent restart loops
if [ -n "$RUNPOD_POD_ID" ]; then
  touch "$SINGLE_RUN_SENTINEL"
  echo "Created sentinel file to prevent restart: $SINGLE_RUN_SENTINEL"
fi

# Self-terminate the RunPod pod if running on RunPod
if [ -n "$RUNPOD_POD_ID" ] && [ -n "$RUNPOD_API_KEY" ]; then
  echo "Self-terminating RunPod pod: $RUNPOD_POD_ID"
  
  # Check if curl is available
  if ! command -v curl &> /dev/null; then
    echo "⚠ Warning: curl not found, cannot send termination API call"
    echo "  Pod will exit but may restart. Install curl in Dockerfile to fix."
  else
    TERMINATE_RESPONSE=$(curl -s -X POST https://api.runpod.io/graphql \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}")
    
    if echo "$TERMINATE_RESPONSE" | grep -q '"podTerminate"'; then
      echo "✓ Pod termination request sent successfully"
    else
      echo "⚠ Pod termination may have failed. Response: $TERMINATE_RESPONSE"
      echo "  Pod may need manual termination"
    fi
  fi
fi

# Exit with training status
# On RunPod, always exit 0 to prevent restart loops even on training failure
if [ -n "$RUNPOD_POD_ID" ]; then
  exit 0
else
  exit $TRAIN_EXIT_CODE
fi
