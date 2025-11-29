# ☁️ Cloud Training & HPO Pipeline

This page documents the cloud-based training and hyperparameter optimization (HPO) workflow for Aegear, leveraging RunPod and ClearML for scalable, automated model development.

---

## 🚀 Overview

Aegear supports launching training jobs and HPO experiments in the cloud using:
- **RunPod**: GPU cloud provider for scalable training.
- **ClearML**: Experiment tracking, orchestration, and HPO management.

This setup enables:
- Automated training job launches with custom configuration.
- Hyperparameter sweeps and grid search for model optimization.
- Experiment tracking and result analysis via ClearML.

---

## 🛠️ Key Scripts & Components


- `tools/launch_runpod_training.py`: Launches a single training job on RunPod with custom arguments and environment variables. **ClearML integration is optional**—if you skip the `--task-name` argument, ClearML tracking will be disabled and the job will run standalone.
- `tools/clearml_runpod_hpo.py`: Orchestrates HPO runs, launching multiple training jobs with different hyperparameters, tracking results in ClearML. **HPO pods are automatically shut down after completion** to minimize cloud costs.
- `docker/run_training.sh`: Entrypoint script used inside the container for reproducible training runs.
- `tools/train.py`: Main training script, used for both local and cloud jobs.

---

## 📦 How It Works

1. **Prepare Docker Image**
   - Use the published image (`ljubobratovicrelja/aegear:latest`) or build your own.
2. **Configure Training**
   - Set environment variables or CLI arguments for your training job (see [docker.md](docker.md)).
3. **Launch Training on RunPod**
   - Use `launch_runpod_training.py` to start a pod with your configuration.
   - Example:
     ```bash
     python tools/launch_runpod_training.py --task-name my_exp --model-type efficient_unet --data-manifest /workspace/data/manifest.json --model-dir /workspace/models/unet --checkpoint-dir /workspace/models/unet/checkpoints --epochs 10 --batch-size 128
     ```
4. **Run HPO Experiments**
    - Use `clearml_runpod_hpo.py` to launch a grid search or sweep over hyperparameters. HPO pods are automatically shut down after each run.
    - Example:
       ```bash
       python tools/clearml_runpod_hpo.py --config config/hpo_config.yaml
       ```
    - See an example HPO YAML config file [here](https://github.com/ljubobratovicrelja/aegear/blob/main/config/hpo_config.yaml).
5. **Track Results in ClearML**
   - All jobs and experiments are tracked in your ClearML dashboard for analysis and comparison.

---

## 🔑 Environment & Credentials

- **RunPod API Token**: Required for launching pods.
- **ClearML Credentials**: Needed for experiment tracking and HPO.
- **Docker Hub Credentials** (optional): For authenticated image pulls.

Set these as environment variables before launching jobs.

---

## 📥 Data Access

Training data shards are available from:
```
gs://aegear-training-data/shards
```

---

## 📖 References & Further Reading
- [RunPod Documentation](https://docs.runpod.io/)
- [ClearML Documentation](https://clear.ml/docs/)
- [Aegear Docker Usage](docker.md)
- [Aegear Training Workflow](training.md)

---

For questions or issues, contact the maintainer or open an issue.
