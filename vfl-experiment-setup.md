# Vertical Federated Learning (VFL) Pipeline — SmallNN

This repository implements a **Vertical Federated Learning (VFL) pipeline** using a simple neural network architecture (`SmallNN`) for clients and a server model for aggregation. It supports **IID and Non-IID client data splits**, tracks utility and fairness metrics, and visualizes client and global performance.

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Models](#models)
* [Training Procedure](#training-procedure)
* [Experiment Setup](#experiment-setup)
* [Metrics Collected](#metrics-collected)
* [Visualization](#visualization)
* [Requirements](#requirements)
* [Usage](#usage)

---

## Overview

This pipeline simulates a **5-client VFL scenario** with a fixed SmallNN for clients and a server neural network. The pipeline evaluates performance under different data partitioning strategies:

* **IID**: Equal number of samples per client
* **Non-IID**: Dirichlet distribution (α=0.5) per class

The pipeline performs **3 global training rounds**, tracking **utility, fairness, and client metrics**.

---

## Dataset

* Preprocessed medical dataset: `medical_processed.csv`
* Features: Auto-detected from CSV
* Target: Binary classification (2 classes)
* Clients: 5
* Data splits: IID, Non-IID

---

## Models

### Client Model (SmallNN)

* Linear layer: `input_dim → hidden_dim=32`
* Activation: ReLU
* Sends activations to server
* Optimizer: SGD, lr=0.01

### Server Model (ServerNN)

* Input: concatenated client activations (`hidden_dim × num_clients = 32 × 5 = 160`)
* Layers: Linear(160 → 16) → ReLU → Linear(16 → 2)
* Loss: CrossEntropyLoss
* Optimizer: SGD, lr=0.01

---

## Training Procedure

1. Load `medical_processed.csv` dataset.
2. Split data into IID and Non-IID client datasets.
3. Initialize **5 client SmallNNs** and **1 server ServerNN**.
4. Train for **3 global rounds** using synchronized batch updates:

   * Clients compute local activations → sent to server
   * Server forward + backward → gradients propagate back to clients
5. Evaluate **local client metrics**, **global accuracy per round**, **test set metrics**, and **fairness metrics (Age)**.

---

## Experiment Setup

| Parameter         | Values / Description           |
| ----------------- | ------------------------------ |
| Number of clients | 5                              |
| Client model      | SmallNN, hidden_dim=32         |
| Server model      | ServerNN, hidden_dim=16        |
| Global rounds     | 3                              |
| Batch size        | 64                             |
| Learning rate     | 0.01 (Client & Server)         |
| Loss              | CrossEntropyLoss               |
| Data splits       | IID, Non-IID (Dirichlet α=0.5) |
| Optimizer         | SGD                            |

**Total experiments:** 2 (IID vs Non-IID)

---

## Metrics Collected

* **Utility Metrics:** Accuracy, F1-score, Precision, Recall
* **Fairness Metrics (Age):** SPD (Statistical Parity Difference), EOD (Equal Opportunity Difference)
* **Client Metrics:** Local accuracy per client
* **Global Metrics:** Global accuracy per round
* **Cost Metrics:** Number of clients, number of global rounds

---

## Visualization

The pipeline generates **multi-panel plots**:

1. Global accuracy per round
2. Client-level accuracy
3. Client sample sizes
4. Fairness metrics (SPD, EOD)

---

## Requirements

* Python 3.8+
* PyTorch
* NumPy, Pandas
* Matplotlib
* scikit-learn
* fluke (for dataset utilities)

---

## Usage

```bash
# Clone repository
git clone <repo_url>
cd <repo_dir>

# Run VFL pipeline (IID & Non-IID)
python run_vfl_pipeline.py
```

* Outputs include:

  * Trained client and server models
  * Global and client metrics per round
  * Plots for accuracy, sample sizes, and fairness

---

**Note:** This pipeline is intended for **small-scale experiments** with a fixed SmallNN architecture. Hyperparameters can be modified in `run_vfl_pipeline.py`.

---

