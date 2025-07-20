# 🐳 Docker Support

Aegear provides a ready-to-use Docker image to simplify deployment and training in cloud environments or on systems without a configured Python environment.

---

## 📦 Public Docker Image

The official Aegear Docker image is hosted on Docker Hub:  

```
ljubobratovicrelja/aegear:latest
```

Pull the image with:  

```bash
docker pull ljubobratovicrelja/aegear:latest
```

---

## 🚀 Running the Image

Launch a container with GPU support (if available):  

```bash
docker run --gpus all -it ljubobratovicrelja/aegear:latest
```

To mount a local directory for accessing datasets or saving models:  

```bash
docker run --gpus all -it -v /path/to/data:/workspace/data ljubobratovicrelja/aegear:latest
```

This will mount `/path/to/data` on your host to `/workspace/data` inside the container.

---

## 🛠 Custom Training with Docker

The public image includes all dependencies for running Aegear and training on the provided dataset. For custom datasets or workflows:  

1. **Clone and Extend**  
   Clone the Aegear repository and modify the Dockerfile as needed:  
   ```bash
   git clone https://github.com/ljubobratovicrelja/aegear.git
   cd aegear
   docker build -t aegear-custom .
   ```

2. **Custom Parameters**  
   When launching the container, you can specify:  
   - **Which notebook** to use (`training_unet.ipynb` or `training_siamese.ipynb`)
   - **Which branch** of the repository (defaults to `main`)  
   
   Example:  
   ```bash
   docker run --gpus all -e NOTEBOOK=training_siamese.ipynb -e BRANCH=dev -it ljubobratovicrelja/aegear:latest
   ```

   > 📌 Currently, this works for the main repository. Support for forks can be added easily by adjusting the Dockerfile’s repository source.

3. **Mount Your Data**  
   Mount your dataset and configuration files into the container using `-v`.

4. **Run Training Scripts**  
   Inside the container, launch Jupyter notebooks or Python scripts from the `notebooks/` directory:  
   ```bash
   jupyter notebook notebooks/training_unet.ipynb
   jupyter notebook notebooks/training_siamese.ipynb
   ```

---

## 📖 Notes

- **GPU Support**: Requires NVIDIA drivers and Docker with NVIDIA runtime (`nvidia-docker2`).
- **Cloud Deployment**: Compatible with Google Cloud, AWS, and Azure GPU instances.
- **Extending**: For additional libraries or dependencies, create a new Dockerfile based on `ljubobratovicrelja/aegear:latest`.

---

## 📦 Source Dockerfile

The Dockerfile used to build the public image is included in the repository at:  
```
aegear/Dockerfile
```

This can serve as a starting point for creating your own custom images.