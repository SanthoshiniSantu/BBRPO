# JWG-GUARD: Jellyfish Search Waterwheel Plant Optimization Algorithm-based Defending Graph Neural Networks Against Adversarial

Attacks

## Overview

This project implements a graph-based adversarial attack detection framework on multiple benchmark datasets:

- CIFAR-10 Superpixel Graphs
- MNIST Image Graphs
- MUTAG Molecular Graphs

The framework extracts graph edge features, generates adversarial edge perturbations, and evaluates several GNN-based defense models.

The proposed model combines:

- Jellyfish Search Optimization (JSO)
- Walrus Optimization Algorithm (WWPA)
- GNNGuard

to form the proposed:

**JWG-GUARD**

framework.

---

## Project Structure

```text
code/
│
├── GNN/
│   └── gnn.py
│
├── GNN_JSO/
│   └── gnn.py
│   └── jellyFish.py
│
├── GNN_WWPA/
│   └── gnn.py
│   └── water_wheel.py
│
├── Proposed_JWG_GUARD/
│   └── gnn.py
│   └── JWWPO.py
│
├── feature_extraction.py
│
├── Run_CIFAR.py
├── Run_MNIST.py
├── Run_MUTAG.py
│
├── Run_ablation_analysis.py
├── Run_main.py
│
├── requirements.txt
│
└── README.md
```


## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Main Experiment

```bash
python Run_main.py
```

---

### CIFAR

```bash
python Run_CIFAR.py
```

---

### MNIST

```bash
python Run_MNIST.py
```

---

### MUTAG

```bash
python Run_MUTAG.py
```

---

### Ablation Study

```bash
python Run_ablation_analysis.py
```

---

## Example Execution

```text
Available Datasets:

1. CIFAR
2. MNIST
3. MUTAG

Enter Dataset : 1

Enter Training Percentage : 80
```

or

```text
Enter K-Fold Value : 5
```

---
## Hardware Requirements

| Component | Requirement |
|------------|-------------|
| Processor | Intel, AMD, or Equivalent Multi-Core Processor |
| RAM | Minimum 8 GB (16 GB Recommended) |
| Storage | Minimum 10 GB Free Disk Space |
| GPU | NVIDIA/AMD GPU (Optional) |

---
## Software Requirements

| Software | Version |
|-----------|----------|
| Operating System | Windows |
| Python | 3.9.11 |