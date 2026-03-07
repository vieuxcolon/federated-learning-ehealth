# Understanding Runs vs Epochs and Data Distribution in Federated Learning

In traditional **centralized machine learning**, training terminology differs from **federated learning (FL)**. This document clarifies key concepts and explains how data is distributed across clients for experiments.

---

## 1. Centralized Training

| Term                | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| **Epoch**           | One full pass over the **entire training dataset** by the model. |
| **Batch/Iteration** | One mini-batch update during training.                           |

**Example:**
If the dataset has 10,000 samples and batch size is 100, one epoch contains 100 iterations (batches).

---

## 2. Federated Learning (FL) Training

In FL, training is **distributed across multiple clients** and coordinated by a central server. Correct terminology helps interpret experiments properly.

| Term                           | Description                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Round (Global Round / Run)** | One communication cycle between the **server** and selected **clients**. Each round consists of: <br> 1. Server selects a subset of clients (based on `client_ratio`). <br> 2. Clients perform **local training** for several **local epochs** on their own data. <br> 3. Clients send **model updates** to the server. <br> 4. Server aggregates updates (e.g., using FedAvg) to update the **global model**. |
| **Local Epoch**                | Number of passes each client makes over **its local dataset** during a single round.                                                                                                                                                                                                                                                                                                                           |
| **Iteration / Mini-batch**     | Mini-batch updates performed on the client’s local dataset.                                                                                                                                                                                                                                                                                                                                                    |

**Key differences vs centralized training:**

| Concept     | Centralized ML    | Federated Learning            |
| ----------- | ----------------- | ----------------------------- |
| Epoch       | Full dataset pass | Local training pass on client |
| Round / Run | Not applicable    | One global aggregation cycle  |
| Iteration   | Mini-batch update | Mini-batch update on client   |

**Example FL log message:**

```text
Running experiment 1: mode=federation, dataset=mnist, client_ratio=0.3, local_epochs=5, distribution=iid
Per-Round Durations: {1: 2.34, 2: 2.56, 3: 2.48, ...}
```

---

## 3. Data Distribution Across Clients

In FL, the **dataset is partitioned across clients**, and the choice of partitioning affects model training and performance.

**Scenario:** Dataset with **100,000 samples**, **1 server**, **5 clients**.

### 3.1 IID (Independent and Identically Distributed)

* Each client receives a **random and balanced share** of the dataset.
* Each client’s data reflects the **global distribution**.

**Example:** Global dataset 75% class 0, 25% class 1

| Client | Class 0 | Class 1 | Total samples |
| ------ | ------- | ------- | ------------- |
| 1      | 15,000  | 5,000   | 20,000        |
| 2      | 15,000  | 5,000   | 20,000        |
| 3      | 15,000  | 5,000   | 20,000        |
| 4      | 15,000  | 5,000   | 20,000        |
| 5      | 15,000  | 5,000   | 20,000        |

---

### 3.2 Dirichlet (Non-IID / Skewed Distribution)

* A **Dirichlet distribution** is used to **skew class proportions** across clients.
* Each client may have **different proportions of each class**, simulating real-world heterogeneity.

**Adult dataset example:**

* Task: binary classification

  * Class 0 → `<=50K`
  * Class 1 → `>50K`

* Dirichlet partition with α = 0.5:

| Client | Class 0 (`<=50K`) | Class 1 (`>50K`) | Total samples |
| ------ | ----------------- | ---------------- | ------------- |
| 1      | 10,000            | 2,000            | 12,000        |
| 2      | 5,000             | 7,000            | 12,000        |
| 3      | 8,000             | 4,000            | 12,000        |
| 4      | 2,000             | 10,000           | 12,000        |
| 5      | 12,000            | 0                | 12,000        |

* Smaller α → more skewed clients; larger α → more balanced clients.
* This creates **heterogeneous client data**, which can affect global model performance and convergence.

**Summary Visual Example:**

```text
IID:       Client 1: 20% each class
           Client 2: 20% each class
           ...
Dirichlet: Client 1: 50% class 0, 10% class 1, ...
           Client 2: 5% class 0, 60% class 1, ...
```

**Key Points:**

* **IID:** Each client mirrors global distribution.
* **Dirichlet:** Clients have uneven class distributions, simulating real-world heterogeneity.
* **Server never sees raw client data**, only aggregated updates.

---

### 4. Recommended Terminology for FL Experiments

* Use **“round”** for global server-client cycles.
* Keep **local epochs** for passes over a client’s own data.
* Record **distribution type** (IID or Dirichlet α) for clarity.

**Example log for Adult dataset with Dirichlet=0.5:**

```text
Running experiment 1: mode=federation, dataset=adult, client_ratio=0.3, local_epochs=5, distribution=dirichlet=0.5
Per-Round Durations: {1: 2.34, 2: 2.56, 3: 2.48, ...}
```
