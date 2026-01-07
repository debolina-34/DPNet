# DPNet
# DPNet: Depthwise–Pointwise CNN for Audio Classification

This repository contains **DPNet**, a lightweight **Depthwise–Pointwise Convolutional Neural Network (CNN)** designed for efficient audio classification, with a focus on **resource-constrained and TinyML-oriented deployments**.

The work includes experiments on **original and amplitude-shifted audio data**, making the model robust to real-world variations in signal intensity.

---

## Motivation

Conventional CNNs are often computationally expensive for embedded platforms.  
DPNet leverages **depthwise separable convolutions** to significantly reduce:

- Model size
- Number of parameters
- Inference latency

while maintaining competitive classification performance.

---

## Model Overview

- **Architecture**: Depthwise–Pointwise CNN (DPNet)
- **Input**: Audio features extracted from raw waveforms  
- **Key Design Goals**:
  - Low computational complexity
  - Robustness to amplitude variations
  - Suitability for TinyML / edge devices

---

## YouTube Link

https://youtu.be/QqQ6c4FGObU


