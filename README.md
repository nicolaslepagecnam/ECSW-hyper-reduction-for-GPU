# GPU-Accelerated ECSW Hyper-Reduction: Implementation and Performance Evaluation

Code accompanying the paper:

> **GPU-Accelerated ECSW Hyper-Reduction: Implementation and Performance Evaluation**

This repository contains the complete computational pipeline for GPU-accelerated Energy-Conserving Sampling and Weighting (ECSW) hyper-reduction, including:
- Finite element simulations in **FreeFem++**
- **POD–Galerkin reduced-order models (ROMs)**
- **ECSW hyper-reduced ROMs**
- **GPU-accelerated implementations using PyTorch**

**Test case:** Fluidic pinball at Reynolds number Re = 30

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## System Requirements

### Operating System
- Linux (tested and recommended)

### FreeFem++
- **Version:** 4.6 or higher
  ```
  FreeFem++ v4.6 (Thu Apr 2 15:47:38 CEST 2020 - git v4.6)
  ```

### Python Environment
- **Python:** 3.11

### CUDA/GPU Stack
- `cudatoolkit`: 11.7.0
- `cuda-runtime`: 11.7.1
- `cuda-cudart`: 11.7.99
- **PyTorch:** 2.0.1 with CUDA 11.7 and cuDNN 8.5.0
  - Installation: `pytorch 2.0.1 py3.11_cuda11.7_cudnn8.5.0_0`

### Required Python Libraries

Core dependencies:
```
numpy==1.26.0
scipy==1.11.3
matplotlib==3.8.0
```

Optional (for GPU/deep-learning experiments):
```
pytorch-lightning==2.1.1
wandb==0.15.12
torchdiffeq==0.2.2
adabelief-pytorch==0.2.1
```

---

## Installation

### 1. Install FreeFem++

Install FreeFem++ version 4.6 or higher following the instructions at [https://freefem.org/](https://freefem.org/)

### 2. Set up Python Environment

Create and activate a new conda environment:

```bash
conda create -n ecsw python=3.11
conda activate ecsw
```

### 3. Install PyTorch with CUDA Support

```bash
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 4. Install Core Dependencies

```bash
pip install numpy==1.26.0 scipy==1.11.3 matplotlib==3.8.0
```

### 5. Install Optional Dependencies (for advanced experiments)

```bash
pip install pytorch-lightning==2.1.1 wandb==0.15.12 torchdiffeq==0.2.2 adabelief-pytorch==0.2.1
```

### 6. Clone This Repository

```bash
git clone https://github.com/yourusername/gpu-ecsw-hyperreduction.git
cd gpu-ecsw-hyperreduction
```

---

## Pipeline Overview

The computational pipeline consists of five main stages:

1. **FEM Simulation (FreeFem++):** Generate high-fidelity ground truth data and export operators
2. **ROM Construction (Python):** Build standard POD-Galerkin reduced-order models
3. **ECSW Hyper-Reduction (Python):** Compute ECSW sampling and weights for efficient evaluation
4. **GPU Implementation (PyTorch):** Accelerate ECSW computations on GPU
5. **Performance Benchmarking:** Evaluate wall-clock time and computational efficiency

---

## Usage

Execute the following scripts in order to reproduce the results from the paper.

### Stage 1: FreeFem++ Simulations

Run these scripts to generate the finite element dataset and operators:

```bash
# 1. Generate computational mesh for fluidic pinball
FreeFem++ mesh_pinball.edp

# 2. Compute steady-state solutions from Re=10 to Re=30 using Newton method
FreeFem++ newton_Re_10_to_30.edp

# 3. Initialize simulation at Re=30
FreeFem++ freefem_init_RE_30.edp

# 4. Compute eigenvalues for stability analysis
FreeFem++ eigen_value_solve.edp

# 5. Run transient direct numerical simulation at Re=30
FreeFem++ dns_transient_RE_30.edp

# 6. Export FEM operators and matrices for ROM construction
FreeFem++ Export_operators_freefem.edp

# 7. Tensorize Gauss quadrature points for ECSW
FreeFem++ tensorization_gauss_points.edp
```

### Stage 2: Standard ROM Construction (CPU)

```bash
# Run I/O utilities (no standalone execution needed, imported by other scripts)
# python funcIO.py

# Initialize and write ROM data structures
python InitWrite.py

# Compute POD basis at Re=30
python ComputePOD_30.py

# Assemble reduced-order model matrices
python ComputeROM_30.py

# Simulate and validate standard ROM
python playROM_30.py
```

### Stage 3: ECSW Hyper-Reduction (CPU)

```bash
# Compute ECSW sample points and weights
python Compute_ECSW_30.py

# Run and validate ECSW hyper-reduced model
python PlayECSW_30.py
```

### Stage 4: GPU-Accelerated ECSW

```bash
# Execute GPU-accelerated ECSW simulation using PyTorch
python torch_GPU_ECSW.py
```

### Stage 5: Performance Benchmarking

```bash
# Wall-clock time evaluation (coming soon)
# python benchmark_performance.py
```

**Note:** Performance benchmarking scripts use placeholder random matrices to isolate computational costs from I/O overhead.

---

## File Descriptions

### FreeFem++ Scripts

| File | Description |
|------|-------------|
| `mesh_pinball.edp` | Generates the computational mesh for the fluidic pinball geometry |
| `newton_Re_10_to_30.edp` | Computes steady-state solutions across Reynolds numbers using Newton's method |
| `freefem_init_RE_30.edp` | Initializes the simulation at Re=30 |
| `eigen_value_solve.edp` | Performs eigenvalue analysis for stability characterization |
| `dns_transient_RE_30.edp` | Executes time-dependent DNS and generates snapshot database |
| `Export_operators_freefem.edp` | Exports mass, stiffness, and nonlinear operators for ROM |
| `tensorization_gauss_points.edp` | Processes Gauss quadrature data for hyper-reduction |

### Python Scripts

#### ROM Construction

| File | Description |
|------|-------------|
| `funcIO.py` | I/O utility functions for data import/export |
| `InitWrite.py` | Initializes ROM data structures |
| `ComputePOD_30.py` | Computes POD basis from DNS snapshots |
| `ComputeROM_30.py` | Assembles projected ROM operators |
| `playROM_30.py` | Time-integrates and validates the standard ROM |

#### ECSW Hyper-Reduction

| File | Description |
|------|-------------|
| `Compute_ECSW_30.py` | Computes ECSW sample indices and weights via NNLS |
| `PlayECSW_30.py` | Time-integrates and validates the ECSW hyper-reduced ROM |

#### GPU Implementation

| File | Description |
|------|-------------|
| `torch_GPU_ECSW.py` | GPU-accelerated ECSW implementation using PyTorch |

---

*Last updated: [Current Date]*