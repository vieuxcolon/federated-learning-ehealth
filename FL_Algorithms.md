# Federated Learning Algorithms

This document lists common **Federated Learning (FL)** algorithms beyond the classical **FedAvg** baseline.
The methods are grouped by their main research goal such as handling **data heterogeneity, personalization, communication efficiency, or robustness**.

---

# 1. Baseline Federated Optimization

These algorithms are the earliest FL methods and serve as the foundation for many later approaches.

| Algorithm  | Description                                                                                                                        |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **FedSGD** | Clients compute gradients locally and send them to the server for aggregation. Communication-heavy but simple baseline.            |
| **FedAvg** | Clients perform several local training steps and send model weights to the server for averaging. The most widely used FL baseline. |

---

# 2. Heterogeneity-Aware Algorithms

Designed to handle **non-IID data distributions across clients**, which is a major challenge in federated learning.

| Algorithm    | Description                                                                              |
| ------------ | ---------------------------------------------------------------------------------------- |
| **FedProx**  | Adds a proximal regularization term to limit divergence between local and global models. |
| **SCAFFOLD** | Uses control variates to correct client drift caused by heterogeneous data.              |
| **FedDyn**   | Introduces dynamic regularization to stabilize federated optimization.                   |
| **FedNova**  | Normalizes client updates to eliminate bias from varying local training steps.           |

---

# 3. Server-Side Optimizer Variants

These methods replace simple averaging with **adaptive optimization methods at the server**.

| Algorithm      | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| **FedOpt**     | General framework applying server-side optimizers.                   |
| **FedAdam**    | Uses the Adam optimizer on aggregated client updates.                |
| **FedYogi**    | Variant of Adam designed for better stability in federated settings. |
| **FedAdagrad** | Uses Adagrad-style adaptive learning rates on the server.            |

---

# 4. Personalized Federated Learning (PFL)

These algorithms aim to produce **client-specific models** instead of a single global model.

| Algorithm      | Description                                                       |
| -------------- | ----------------------------------------------------------------- |
| **pFedMe**     | Uses bi-level optimization to personalize models for each client. |
| **Ditto**      | Trains both global and personalized models simultaneously.        |
| **Per-FedAvg** | Meta-learning based approach for quick personalization.           |
| **FedRep**     | Shares representations globally but keeps local classifier heads. |
| **FedBN**      | Keeps batch normalization layers client-specific.                 |

---

# 5. Representation-Based Federated Learning

These methods focus on improving **feature representation learning** in federated environments.

| Algorithm | Description                                                                    |
| --------- | ------------------------------------------------------------------------------ |
| **MOON**  | Uses model-level contrastive learning to align representations across clients. |
| **FedLC** | Corrects label distribution skew during training.                              |

---

# 6. Asynchronous Federated Learning

These algorithms remove the requirement that all clients must synchronize every round.

| Algorithm    | Description                                                                   |
| ------------ | ----------------------------------------------------------------------------- |
| **FedAsync** | Allows asynchronous client updates to the server.                             |
| **FedBuff**  | Buffers client updates before aggregation to stabilize asynchronous training. |
| **FedASGD**  | Asynchronous stochastic gradient descent in federated settings.               |

---

# 7. Robust Aggregation Algorithms

Designed to defend against **malicious or unreliable clients**.

| Algorithm              | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| **Krum**               | Selects updates closest to the majority to resist Byzantine attacks.         |
| **Trimmed Mean**       | Removes extreme values before aggregation.                                   |
| **Median Aggregation** | Uses coordinate-wise median instead of mean to reduce influence of outliers. |

---

# 8. Other Specialized Federated Learning Algorithms

These methods address specific issues such as fairness or data imbalance.

| Algorithm                             | Description                                                              |
| ------------------------------------- | ------------------------------------------------------------------------ |
| **AFL (Agnostic Federated Learning)** | Focuses on fairness across client distributions.                         |
| **Astraea**                           | Client sampling strategy designed to mitigate data imbalance.            |
| **HyFDCA**                            | Hybrid federated learning supporting multiple data partitioning schemes. |

---

# Summary

Federated learning algorithms can generally be categorized into the following groups:

* **Baseline optimization:** FedSGD, FedAvg
* **Heterogeneity-aware:** FedProx, SCAFFOLD, FedDyn, FedNova
* **Server optimizer variants:** FedAdam, FedYogi, FedAdagrad
* **Personalized FL:** pFedMe, Ditto, Per-FedAvg, FedRep
* **Representation learning:** MOON, FedLC
* **Asynchronous FL:** FedAsync, FedBuff
* **Robust aggregation:** Krum, Trimmed Mean, Median

---

