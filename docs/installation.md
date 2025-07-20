# Installation

Aegear can be used either as a Python library for advanced workflows or as a standalone graphical user interface (GUI) application. Choose the method that best suits your needs.

---

## 🚀 GUI Application (Recommended for Most Users)

Pre-built standalone binaries for Windows and macOS are available on the [GitHub Releases page](https://github.com/ljubobratovicrelja/aegear/releases). These executables are packaged using PyInstaller and do not require installation.

### 🖥️ Windows
1. Download the latest Windows binary:
   - [aegear-gui-v0.4.1-win64-cpu.exe](https://github.com/ljubobratovicrelja/aegear/releases/download/v0.4.1/aegear-gui-v0.4.1-win64-cpu.exe)
2. Place the downloaded file in a folder of your choice.
3. Double-click to run Aegear directly. No installation is required.

> ⚠️ **CUDA Support**:  
> The Windows binary includes CPU-only PyTorch.  
> To use GPU acceleration, install Aegear from source (see below).

---

### 🍏 macOS
1. Download the macOS (Apple Silicon) binary:
   - [aegear-gui-v0.4.1-mac-arm64.zip](https://github.com/ljubobratovicrelja/aegear/releases/download/v0.4.1/aegear-gui-v0.4.1-mac-arm64.zip)
2. Unzip the archive.
3. Double-click the Aegear app to run it. You may need to right-click → Open on first launch to bypass Gatekeeper restrictions.

---

## 🔧 Python Library (For Developers)

If you want to use Aegear programmatically or with GPU support, install from source:

### 1. Clone the Repository
```bash
git clone https://github.com/ljubobratovicrelja/aegear.git
cd aegear
```

### 2. Install Dependencies
It’s recommended to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install -e .[dev]
```

This installs Aegear in editable mode along with developer tools (notebooks, training).

---

### 3. Launch the GUI
If you installed the Python package, you can launch the GUI directly:
```bash
aegear-gui
```

---

## 📌 Requirements

| Component        | Minimum Version |
|------------------|-----------------|
| Python           | 3.10+           |
| PyTorch          | 2.6+            |
| CUDA (optional)  | 12.1+           |

---

## 📖 Documentation
For more details, see [Usage](usage.md) and [API Reference](api.md).