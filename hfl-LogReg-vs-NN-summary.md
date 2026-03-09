---

# Horizontal Federated Learning (HFL) with a Medical Dataset

This repository demonstrates **Horizontal Federated Learning (HFL)** on a medical dataset using **two model pipelines**: a small PyTorch Neural Network (SmallNN) and Logistic Regression (LR). Both pipelines implement IID and Non-IID experiments and include multi-panel visualizations for performance, fairness, and client analysis.

---

## 1. Horizontal Federated Learning Logical Flow

```mermaid
flowchart TD
    A[Load Processed Dataset] --> B[Split Data Across Clients]
    B --> C[Initialize Global Model]
    C --> D[For Each Global Round]
    D --> E[Send Global Model to Clients]
    E --> F[Clients Train Local Model on Local Data]
    F --> G[Clients Send Updated Model Parameters]
    G --> H[Server Aggregates Models (FedAvg)]
    H --> I[Update Global Model]
    I --> D
    I --> J[Evaluate Global Model on Test Set]
    J --> K[Compute Performance Metrics (Accuracy, F1, Precision, Recall)]
    J --> L[Compute Fairness Metrics (SPD, EOD)]
    K --> M[Generate Multi-panel Plots]
    L --> M
```

> The diagram above shows the HFL pipeline workflow: client-local training, federated averaging, evaluation, and visualization.

---

## 2. Pipeline 1: Small PyTorch Neural Network (SmallNN)

### 2.1 Pipeline Overview

1. **Data Loading:** Load preprocessed medical dataset from CSV
2. **Client Splitting:** IID or Non-IID (Dirichlet)
3. **Model:** Feedforward NN with one hidden layer (16 neurons)
4. **Training:** Each client trains locally per global round; FedAvg aggregates weights
5. **Evaluation:** Test set evaluation + client-level accuracy
6. **Fairness Metrics:** SPD & EOD by age group
7. **Visualization:** Multi-panel plots: global vs client accuracy, client sample sizes, fairness metrics, global accuracy per round

### 2.2 Results Summary

**IID Experiment**

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 0.6639  |
| F1 Score  | 0.7979  |
| SPD (Age) | -0.0001 |
| EOD (Age) | -0.0030 |

**Non-IID Experiment**

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 0.6637  |
| F1 Score  | 0.7979  |
| SPD (Age) | -0.0000 |
| EOD (Age) | -0.0029 |

**Client-level Accuracy Comparison (final round)**

| Client | IID    | Non-IID |
| ------ | ------ | ------- |
| 1      | 0.6685 | 0.7174  |
| 2      | 0.6661 | 0.9770  |
| 3      | 0.6669 | 0.5180  |
| 4      | 0.6655 | 0.5293  |
| 5      | 0.6663 | 0.9762  |

### 2.3 Multi-Panel Visualization (SmallNN)

![SmallNN HFL Multi-panel](figures/smallnn_hfl_multiplot.png)

> Top-left: Global Accuracy per Round
> Top-right: Client-level Accuracy Comparison
> Bottom-left: Client Sample Sizes
> Bottom-right: Fairness Metrics (SPD & EOD by Age)

---

## 3. Pipeline 2: Logistic Regression (LR)

### 3.1 Pipeline Overview

* Same dataset and client split strategy as SmallNN
* Logistic Regression trained locally per client
* FedAvg aggregates model coefficients
* Evaluation metrics and fairness computed as in SmallNN
* Multi-panel plots generated to summarize results

### 3.2 Results Summary

**IID Experiment**

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 0.6635  |
| F1 Score  | 0.7976  |
| SPD (Age) | -0.0009 |
| EOD (Age) | -0.0038 |

**Non-IID Experiment**

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 0.6635  |
| F1 Score  | 0.7976  |
| SPD (Age) | -0.0009 |
| EOD (Age) | -0.0038 |

**Client-level Accuracy Comparison (final round)**

| Client | IID    | Non-IID |
| ------ | ------ | ------- |
| 1      | 0.3327 | 0.3757  |
| 2      | 0.6652 | 0.6472  |
| 3      | 0.6671 | 0.9774  |
| 4      | 0.6645 | 0.9935  |
| 5      | 0.6655 | 0.9505  |

### 3.3 Multi-Panel Visualization (LR)

![LR HFL Multi-panel](figures/lr_hfl_multiplot.png)

> Layout is identical to SmallNN: global vs client accuracy, client sample sizes, fairness metrics, global accuracy per round

---

## 4. Comparison: SmallNN vs Logistic Regression

| Aspect                    | SmallNN                   | Logistic Regression      |
| ------------------------- | ------------------------- | ------------------------ |
| Global Accuracy           | 0.6637–0.6639             | 0.6635                   |
| F1 Score                  | 0.7977–0.7979             | 0.7976                   |
| Client Variance (Non-IID) | Large                     | Large                    |
| Fairness Metrics          | Near zero                 | Near zero                |
| Model Complexity          | Hidden layer (16 neurons) | Linear (no hidden layer) |
| Training                  | PyTorch SGD               | scikit-learn / SGD       |

**Observations:**

* Both models achieve comparable global performance despite differences in complexity.
* Non-IID splits increase variance at the client level but global accuracy and fairness remain stable.
* Multi-panel plots provide intuitive visualization of performance, fairness, and client sample distribution.

---

## 5. Conclusion

1. **HFL preserves privacy** while enabling collaborative medical data analysis.
2. **SmallNN and LR pipelines** both provide robust performance for binary medical classification.
3. **Non-IID client distributions** challenge federated learning but do not significantly reduce global model accuracy.
4. **Fairness metrics** (SPD & EOD) remain low across all experiments, indicating equitable predictions across age groups.
5. **Multi-panel visualizations** enhance interpretability of federated results.

> This repository provides a complete HFL workflow: data splitting, local training, federated aggregation, evaluation, fairness analysis, and visualization.

---

**Next Steps / Recommendations:**

* Extend pipelines to more complex medical datasets with multiple features or classes
* Add privacy-preserving techniques (e.g., Differential Privacy, Secure Aggregation)
* Explore hyperparameter tuning for improved global performance

---


