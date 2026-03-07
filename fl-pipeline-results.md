# Federated Learning Experiment Results — Full Analysis

This document summarizes the results of **36 experiments** comparing **centralized learning** and **federated learning** across two datasets and multiple hyperparameter configurations.

Datasets used:

* MNIST
* Adult dataset

Training modes:

* **Centralized training**
* **Federated training**

Federated parameters explored:

* `client_ratio ∈ {0.3, 0.6, 1.0}`
* `local_epochs ∈ {5, 10, 20}`
* `global_rounds = 20`

Total experiments:

```
36
```

---

# 1. Experiment Overview

The pipeline evaluated the effect of **federated training parameters** on model performance and runtime.

Models used:

| Dataset | Model                    |
| ------- | ------------------------ |
| MNIST   | MNIST_2NN neural network |
| Adult   | Logistic Regression      |

Experiment distribution:

| Mode        | Dataset | Experiments |
| ----------- | ------- | ----------- |
| Centralized | MNIST   | 9           |
| Centralized | Adult   | 9           |
| Federated   | MNIST   | 9           |
| Federated   | Adult   | 9           |

Total:

```
36 runs
```

---

# 2. Centralized Training Results

Centralized runs establish the **performance upper bound**.

### MNIST

Dataset:
MNIST

Best result:

```
Accuracy ≈ 0.9803
```

Observations:

* Accuracy remained **stable across all configurations**
* Local epochs had **minimal impact**
* Runtime increased slightly with larger settings

Reason:

Centralized training uses **all data each iteration**, making convergence consistent regardless of configuration.

---

### Adult Dataset

Dataset:
Adult dataset

Best result:

```
Accuracy ≈ 0.8392
```

Observations:

* Stable performance across configurations
* Logistic regression converges quickly
* Runtime remained low

Centralized results serve as the **baseline comparison** for federated learning.

---

# 3. Federated Learning Results (MNIST)

Dataset:

MNIST

Model:

```
MNIST_2NN
```

Federated training introduces two major variables:

```
client_ratio
local_epochs
```

---

# 4. Effect of Client Participation

### client_ratio = 0.3

Accuracy range:

```
0.739 – 0.792
```

Observations:

* Lowest accuracy among configurations
* Slow convergence

Reason:

Only **30% of clients participate per round**, meaning:

* fewer data samples per update
* noisier gradient aggregation

---

### client_ratio = 0.6

Accuracy range:

```
0.814 – 0.853
```

Performance improves significantly.

More clients contribute updates, producing **more stable global gradients**.

---

### client_ratio = 1.0

Accuracy range:

```
0.889 – 0.911
```

Best configuration:

```
client_ratio = 1.0
local_epochs = 20
accuracy ≈ 0.9109
```

Conclusion:

Increasing client participation significantly improves **federated model performance**.

---

# 5. Federated vs Centralized Performance Gap

Best results comparison:

| Mode        | Dataset | Accuracy |
| ----------- | ------- | -------- |
| Centralized | MNIST   | 0.9803   |
| Federated   | MNIST   | 0.9109   |

Performance difference:

```
≈ 7% accuracy drop
```

This gap is expected in federated learning and is described in the original algorithm proposed by
Brendan McMahan.

Common causes:

* distributed data partitions
* limited client participation
* communication delays
* client update variance

---

# 6. Effect of Local Training Epochs

Across all client ratios, increasing **local epochs** improved accuracy.

Example (client_ratio = 1.0):

| Local Epochs | Accuracy |
| ------------ | -------- |
| 5            | 0.889    |
| 10           | 0.906    |
| 20           | 0.911    |

Interpretation:

Longer local training allows clients to compute **stronger parameter updates** before aggregation.

However, excessive local training may cause **client drift**, where models diverge before synchronization.

In these experiments, the regime remained stable.

---

# 7. Runtime Behavior

Federated runtime scaled strongly with local training.

Example:

| Configuration                   | Runtime |
| ------------------------------- | ------- |
| client_ratio = 1.0, epochs = 5  | ~251 s  |
| client_ratio = 1.0, epochs = 10 | ~413 s  |
| client_ratio = 1.0, epochs = 20 | ~764 s  |

Runtime roughly follows:

```
training cost ≈ clients × local_epochs × rounds
```

This demonstrates a key challenge of federated learning:

**communication + computation cost increases quickly with scale.**

---

# 8. Federated Adult Dataset Experiments

Dataset:

Adult dataset

Experiments:

```
Runs 28–36
```

Result:

```
accuracy = None
```

Observed runtime:

```
~30 seconds per run
```

This strongly suggests the experiments **did not execute full federated training**.

Possible reasons:

1. metrics file not generated
2. incorrect metrics path in the parser
3. federation rounds terminated early
4. logging incompatibility with the Adult dataset

Because centralized Adult runs worked correctly, the issue likely lies in **federated experiment logging** rather than the model itself.

---

# 9. Next Step: Try to fix federated Learning for the adult dataset

``
