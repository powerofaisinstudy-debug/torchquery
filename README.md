# TorchQuery 🛡️

<p align="center">
  <img src="https://raw.githubusercontent.com/powerofaisinstudy-debug/torchquery/main/tch.png" width="600" alt="TorchQuery Logo">
</p>

---

[![PyPI version](https://img.shields.io/pypi/v/torchquery.svg)](https://pypi.org/project/torchquery/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TorchQuery** is a high-performance reliability engine for PyTorch. It provides a "Neural Shield" against **Silent Data Corruption (SDC)**, hardware bit-flips, and numerical instability in massive Deep Learning models.

## 🚀 Key Features

* **Billion-Scale Protection:** Optimized streaming logic designed to handle tensors with $10^9$ elements without crashing.
* **Neural Healing:** Automatically detects and repairs corrupted weights or neurons using statistical outlier detection ($\sigma$-clamping).
* **Distributed SyncBatch:** Cluster-aware protection using `All-Reduce` to ensure safety across multi-GPU and multi-server environments.
* **Zero-Invasive:** Simply wrap your existing tensors or model parameters; no architecture changes required.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/powerofaisinstudy-debug/torchquery/main/chl.png" width="600">
</p>

### Visualizing Silent Data Corruption (SDC)

Hardware glitches—like cosmic rays or VRAM overclocks—can cause random bit-flips. These create massive statistical outliers or `NaNs` in your tensor data.

[Image Link to Image_5.png]

**TorchQuery** acts as a `Neural Shield` that sweeps your multidimensional arrays. It identifies values that can lead to exploding gradients (`3e38`) or numerical instability (`NaN`), "healing" them before they propagate.

**Pre-Sweep State:**
* `NaN` (Not a Number): Corrupts entire model during backpropagation.
* `3e38`: Causes exploding gradients, destroying training stability.

**Post-Sweep State:**
* Invalid data is removed, leaving behind validated tensor values.
