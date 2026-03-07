# Data Partitioning in Federated Learning using the Adult Dataset

## Overview

Federated Learning (FL) is a distributed machine learning paradigm where multiple clients collaboratively train a model **without sharing their raw data**. Instead of training on a centralized dataset, the data is **partitioned across several simulated or real clients**, and each client trains a model locally.

This document explains how **data partitioning works in Federated Learning** using the **UCI Adult Dataset**, which is used in this repository’s FL experiments.

---

## 1. Original Dataset

The **UCI Adult Dataset** is a widely used tabular dataset for predicting whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.

Typical dataset characteristics:

| Property          | Value                 |
| ----------------- | --------------------- |
| Task              | Binary classification |
| Target            | `income > 50K`        |
| Raw size          | ~48,842 samples       |
| Preprocessed size | ~27k–32k samples      |

Example features:

| Feature        | Example      |
| -------------- | ------------ |
| age            | 39           |
| workclass      | Private      |
| education      | Bachelors    |
| occupation     | Tech-support |
| hours-per-week | 40           |
| label          | <=50K        |

At this stage the dataset is **centralized**, meaning all samples exist in a single location.

---

## 2. Train/Test Split

Before applying federated partitioning, the dataset is split into:

* **Training dataset**
* **Test dataset**

Example split:

```
Train: 80%
Test : 20%
```

The test dataset **remains centralized** and is used only for evaluating the global model.

```
Adult Dataset
     |
     |--- Train (80%)
     |
     |--- Test  (20%)  → Global evaluation dataset
```

This ensures that model performance is evaluated consistently across training rounds.

---

## 3. Client Simulation

Federated learning assumes the existence of multiple **independent clients** (devices, organizations, or users).

In experiments, these clients are **simulated by partitioning the training data**.

Example configuration:

```
n_clients = 10
```

Conceptually:

```
Client 1
Client 2
Client 3
...
Client 10
```

Each client receives its **own subset of the training data**.

---

## 4. Data Partitioning Strategies

The most important step in FL data preparation is **how the training dataset is divided among clients**.

Two major strategies are commonly used.

---

### 4.1 IID Partitioning

IID stands for **Independent and Identically Distributed**.

Each client receives:

* approximately the same **number of samples**
* a **similar label distribution**

Example:

```
27,000 training samples
10 clients
```

Each client receives approximately:

```
~2700 samples
```

Example distribution:

| Client    | Samples |
| --------- | ------- |
| Client 1  | 2700    |
| Client 2  | 2700    |
| Client 3  | 2700    |
| ...       | ...     |
| Client 10 | 2700    |

Label distribution remains similar across clients:

```
<=50K : ~75%
>50K  : ~25%
```

IID partitioning behaves similarly to **centralized training**, making it easier for models to converge.

---

### 4.2 Non-IID Partitioning (Dirichlet Sampling)

Real-world federated learning systems rarely have IID data.

Different users typically have **different data distributions**.

To simulate this, FL experiments commonly use **Dirichlet sampling**.

```
distribution = Dirichlet(beta)
```

The **beta parameter** controls how uneven the client data distributions become.

| Beta | Behavior               |
| ---- | ---------------------- |
| 10   | Almost IID             |
| 1    | Mild heterogeneity     |
| 0.5  | Moderate heterogeneity |
| 0.1  | Strong heterogeneity   |
| 0.02 | Extreme heterogeneity  |

Example with:

```
beta = 0.5
```

Client dataset sizes may look like:

```
Client 1 → 5000 samples
Client 2 → 3000 samples
Client 3 → 800 samples
Client 4 → 4200 samples
...
```

Label distributions may also differ:

```
Client 1 → mostly <=50K
Client 2 → balanced
Client 3 → mostly >50K
```

This setup better represents **real federated environments**.

---

## 5. Client Local Datasets

After partitioning, each client owns a **local dataset**.

Example:

```
Client 1 dataset
----------------
x1, y1
x5, y5
x22, y22
...

Client 2 dataset
----------------
x3, y3
x9, y9
x101, y101
...

Client 3 dataset
----------------
x7, y7
x33, y33
...
```

Key principle:

```
Clients never share raw data.
```

Only **model updates** are communicated to the central server.

---

## 6. Local Training

At the start of each training round:

1. The **global model** is sent to selected clients.
2. Each client trains the model **using only its local dataset**.

Example process:

```
Global model → distributed to clients

Client 1 trains locally
Client 2 trains locally
Client 3 trains locally
...
```

Each client produces:

```
local_model_weights
```

---

## 7. Model Aggregation

The central server collects the trained client models and aggregates them into a new global model.

The most common algorithm is **Federated Averaging (FedAvg)**.

Aggregation formula:

```
w_global =
    (n1*w1 + n2*w2 + n3*w3 + ... + nk*wk)
    / total_samples
```

Where:

```
wi = model weights from client i
ni = number of samples at client i
```

Clients with more data have a **larger influence on the global model**.

---

## 8. Global Evaluation

After aggregation, the updated global model is evaluated on the **central test dataset**.

```
Test Dataset
----------------
x_test1
x_test2
x_test3
...
```

Typical metrics:

* Accuracy
* Loss
* Precision
* Recall
* F1 score

Example:

```
Round 1  accuracy → 0.72
Round 5  accuracy → 0.80
Round 10 accuracy → 0.84
```

---

## 9. Federated Training Rounds

Federated learning proceeds through multiple communication rounds.

```
for round in 1..R:
    sample clients
    train locally
    aggregate models
    evaluate global model
```

Typical experiment settings:

```
rounds = 20 – 100
clients = 10 – 100
local_epochs = 1 – 20
```

---

## 10. Why Data Partitioning Matters

The partitioning strategy strongly influences:

* model convergence
* final accuracy
* fairness across clients
* system stability

Improper partitioning can cause issues such as:

```
empty clients
training crashes
biased models
unstable convergence
```

For example, extremely small Dirichlet beta values may assign **zero samples to some clients**, which can break training pipelines.

---

## 11. Summary of the FL Data Pipeline

```
Adult Dataset
      |
      |--- Preprocessing
      |
      |--- Train/Test Split
      |
      |--- Federated Partitioning
              |
              |--- Client 1 dataset
              |--- Client 2 dataset
              |--- Client 3 dataset
              ...
              |--- Client N dataset
      |
      |--- Local Training
      |
      |--- Model Aggregation (FedAvg)
      |
      |--- Global Evaluation
      |
      |--- Repeat for multiple rounds
```

This pipeline enables machine learning models to be trained **collaboratively across distributed datasets while preserving data locality and privacy**.

---

## References

* UCI Machine Learning Repository — Adult Dataset
* McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)

