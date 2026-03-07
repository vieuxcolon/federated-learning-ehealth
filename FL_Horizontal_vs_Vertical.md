# Centralized Learning vs Horizontal Federated Learning (HFL) vs Vertical Federated Learning (VFL)

## Overview

This document summarizes the differences between **centralized learning**, **horizontal federated learning (HFL)**, and **vertical federated learning (VFL)** as implemented in **Fluke**, a federated learning experimentation framework.

The repository architecture supports all three training modes:

```
fluke centralized
fluke federated
fluke vertical
```

However, the experiments in this project focus on **vertical federated learning** while still keeping the architecture compatible with the other modes for future comparisons.

---

# 1. Centralized Learning

### Command

```
fluke centralized config/exp.yaml config/alg.yaml
```

### Data Distribution

All data from participating institutions is collected and stored in a **single centralized location**.

Example:

```
Hospital A
Hospital B
Hospital C
Hospital D
      ↓
   Central Server
```

The dataset becomes a standard machine learning dataset:

```
patients × features
```

Example structure:

```
1000 patients
40 features
```

### Training Process

1. Data from all hospitals is merged into one dataset.
2. A single model is trained on the full dataset.
3. No distributed learning occurs.

### Advantages

* Highest possible model performance
* Simplest training pipeline
* No communication overhead

### Disadvantages

* Violates many data privacy regulations
* Requires raw data sharing across institutions

### Role in Experiments

Centralized learning is typically used as an **upper performance bound** for comparison.

Example:

| Method      | Accuracy |
| ----------- | -------- |
| Centralized | 0.91     |
| Federated   | 0.88     |

---

# 2. Horizontal Federated Learning (HFL)

### Command

```
fluke federated config/exp.yaml config/alg.yaml
```

Horizontal federated learning is the **most common federated learning setup**.

### Data Distribution

Each institution holds **different samples (rows)** but **the same feature space (columns)**.

Example:

| Hospital | Patients | Features         |
| -------- | -------- | ---------------- |
| A        | 1–250    | same 40 features |
| B        | 251–500  | same 40 features |
| C        | 501–750  | same 40 features |
| D        | 751–1000 | same 40 features |

Representation:

```
Hospital A → patients 1–250
Hospital B → patients 251–500
Hospital C → patients 501–750
Hospital D → patients 751–1000
```

All hospitals measure **the same variables**.

### Training Process

Typical FL training loop:

1. Server sends the global model to all clients.
2. Each hospital trains the model locally.
3. Clients send model updates to the server.
4. The server aggregates updates (e.g., using FedAvg).
5. The updated global model is redistributed.

This process repeats for multiple **communication rounds**.

### Advantages

* Raw data never leaves the institutions
* Works well when organizations collect similar data

### Disadvantages

* Communication overhead between clients and server
* Slightly lower accuracy than centralized training

### Typical Applications

* multi-hospital collaborations
* distributed mobile device learning
* cross-institution machine learning

---

# 3. Vertical Federated Learning (VFL)

### Command

```
fluke vertical config/exp.yaml config/alg.yaml
```

Vertical federated learning is used when institutions hold **different features of the same entities**.

### Data Distribution

Each institution contains **different columns of the same dataset**.

Example dataset:

```
1000 patients
40 features
```

Vertical partition:

| Hospital | Features         |
| -------- | ---------------- |
| A        | demographics     |
| B        | laboratory tests |
| C        | imaging features |
| D        | clinical history |

Representation:

```
Hospital A → features 1–10
Hospital B → features 11–20
Hospital C → features 21–30
Hospital D → features 31–40
```

All hospitals refer to the **same patient population**.

### Training Process

Instead of sharing model parameters, institutions share **intermediate representations (embeddings)**.

Typical VFL workflow:

1. Each hospital computes local embeddings using its features.
2. Embeddings are sent to the central server.
3. The server concatenates embeddings from all clients.
4. The server computes predictions and loss.
5. Gradients are sent back to the clients.
6. Clients update their local models.

### Advantages

* Raw feature data never leaves institutions
* Enables collaboration when features are distributed across organizations

### Disadvantages

* Higher communication complexity
* Requires entity alignment across institutions
* More complex training pipeline

### Typical Applications

* healthcare + insurance collaborations
* banking + credit scoring partnerships
* cross-industry datasets

---

# Visual Comparison

| Mode          | Data Split | Example            |
| ------------- | ---------- | ------------------ |
| Centralized   | none       | all data merged    |
| Horizontal FL | rows       | different patients |
| Vertical FL   | columns    | different features |

### Centralized

```
patients × features
(all data combined)
```

### Horizontal FL

```
Hospital A → patients 1–250
Hospital B → patients 251–500
Hospital C → patients 501–750
Hospital D → patients 751–1000
```

All institutions share the same features.

### Vertical FL

```
Hospital A → features 1–10
Hospital B → features 11–20
Hospital C → features 21–30
Hospital D → features 31–40
```

All institutions share the same patients.

---

# Relation to This Project

This repository focuses on experiments using:

```
fluke vertical
```

to simulate **multiple hospitals collaboratively training a model where each hospital holds different subsets of medical features**.

The pipeline tracks the following metrics:

### Utility Metrics

* Accuracy
* F1 score
* Recall
* Precision

### Fairness Metrics

* Statistical Parity Difference (SPD)
* Equal Opportunity Difference (EOD)

### Communication Metrics

* communication rounds
* embedding transfer cost

### Performance Metrics

* accuracy vs federated learning rounds

---

# Summary

| Method        | Data Distribution  | Privacy | Complexity |
| ------------- | ------------------ | ------- | ---------- |
| Centralized   | all data merged    | low     | low        |
| Horizontal FL | different samples  | high    | medium     |
| Vertical FL   | different features | high    | high       |

In short:

* **Centralized learning** provides the highest performance but requires full data sharing.
* **Horizontal federated learning** distributes data across institutions by samples.
* **Vertical federated learning** distributes data across institutions by features.

This project focuses on **vertical federated learning** while maintaining compatibility with other training modes within the Fluke framework.
