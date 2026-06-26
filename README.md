

<p align="center">
  <img src="https://raw.githubusercontent.com/powerofaisinstudy-debug/torchquery/main/tch.png" width="600" alt="TorchQuery Logo">
</p>
---

## 🚀 Vectorized SQL/Pandas Engine for Neural Healing & Tensor Integrity

**TorchQuery** is a high-performance, GPU-accelerated tensor query library designed natively for PyTorch. It brings a clean, declarative, SQL/Pandas-style syntax directly to your live machine learning models, allowing you to intercept, query, and surgically heal corrupted weights (`NaN`, `Inf`, Silent Data Corruption) completely on the GPU.

No more dropping back to the CPU to use Pandas. No more complex, multi-line manual boolean masks. 

<p align="center">
  <a href="https://pypi.org/project/torchquery/"><img src="https://img.shields.io/pypi/v/torchquery.svg?style=for-the-badge" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://discuss.pytorch.org/t/introducing-torchquery-vectorized-engine-for-neural-healing-and-tensor-management/224803"><img src="https://img.shields.io/badge/Community-PyTorch%20Forums-FF4500?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Forums"></a>
  <a href="https://github.com/powerofaisinstudy-debug/torchquery"><img src="https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
</p>

---

## 🌐 Quick Links
* 📦 **PyPI Registry:** [pypi.org/project/torchquery](https://pypi.org/project/torchquery/)
* 💬 **Community Discussion:** [Official PyTorch Forums Thread](https://discuss.pytorch.org/t/update-bringing-sql-pandas-style-vectorized-healing-to-torchquery-gpu-colab-ready/225125)
* 🐛 **Bug Tracker:** [Report an Issue / Feature Request](https://github.com/powerofaisinstudy-debug/torchquery/issues)

---

## ⚡ Key Features

* **Zero-Abstraction CUDA Acceleration:** Automatically detects and targets local NVIDIA GPUs or Google Colab T4/A100 instances without requiring you to write raw CUDA C++ scripts.
* **Surgical Neural Healing:** Execute localized weight repairs during live training runs via simple query strings.
* **Silent Data Corruption (SDC) Protection:** Run vectorized data integrity loops to safeguard billion-scale parameters instantly.
* **Unified DataFrame Syntax:** Work with multi-dimensional tensors using intuitive string-based query parsers.

---

## 🛠️ Hardware Setup & Auto-Detection

TorchQuery features a built-in smart initialization loop. If a user runs your code on Google Colab or an NVIDIA-backed machine, the engine instantly maps operations to CUDA. If no GPU is available, it gracefully falls back to CPU.

```python
import torchquery as tq

# Automatically initializes and locks onto your local GPU or Colab instance
device = tq.initialize_device()


💡 Quick Start: SQL-Style Neural HealingSay goodbye to exploded training runs. When a gradient anomaly or hardware instability introduces a NaN weight, patch it instantly on the fly:Pythonimport torch
import torchquery as tq

# 1. Simulate a massive layer weight matrix on an NVIDIA / Colab GPU
weights = torch.randn(1024, 1024, device='cuda')
weights[512, 512] = float('nan')  # Simulated Silent Data Corruption (SDC)

# 2. Perform a vectorized surgical strike directly on the GPU
# This checks, masks, and replaces corrupted spots instantly
healed_weights = tq.heal(weights, where="value == NaN", replace_with=0.0)

print("Tensor healed successfully on device:", healed_weights.device)


---

📈 Performance AdvantageOperationPandas (CPU Baseline)TorchQuery (NVIDIA GPU / Colab)Data TransferForces Device-to-Host (Slow)0ms (Stays Native on GPU)Mask EvaluationSequential / Single-ThreadedParallelized Vector KernelsLarge Tensor ScalingMemory BottleneckBlazing Fast CUDA Streams📦 InstallationBashpip install torchquery
Built with 🧡 for the PyTorch Community. Keep your weights safe, and your training loops stable.
