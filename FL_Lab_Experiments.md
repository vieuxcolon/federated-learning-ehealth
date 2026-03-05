# Federated Learning Mini Research Pipeline

## Overview

This repository contains a **reproducible research pipeline** for running and analyzing **Federated Learning (FL)** experiments using the **Fluke framework**.

The pipeline is designed to automatically execute a grid of experiments comparing:

* **Centralized learning**
* **Federated learning**

across multiple datasets and hyperparameter settings.

The entire workflow runs in **Google Colab** and includes automated experiment execution, metrics collection, result aggregation, and visualization.

---

# Objectives

The goal of this project is to:

* Evaluate **centralized vs federated training performance**
* Study the impact of **client participation ratio**
* Study the effect of **local training epochs**
* Compare behavior across **different datasets**

The pipeline ensures **fully reproducible experiments** by dynamically generating configuration files and automatically executing all combinations.

---

# Methodology

This project evaluates the behavior of **Federated Learning (FL)** under different training conditions using a controlled experimental pipeline.

The methodology follows a **systematic grid-search experiment design** where multiple parameters affecting FL performance are varied while keeping other settings constant.

## 1. Training Paradigms

Two training paradigms are compared:

### Centralized Learning

* All training data is aggregated into a single dataset.
* A single global model is trained directly using standard gradient descent.
* Serves as the **performance upper bound baseline**.

### Federated Learning

* Data remains distributed across multiple simulated clients.
* Each client trains the model locally for several epochs.
* Local models are sent to the server and aggregated using **Federated Averaging (FedAvg)**.

The FL implementation relies on the **FedAvg algorithm**, which performs weighted averaging of model parameters across participating clients.

---

## 2. Parameter Exploration

To understand how federated systems behave under different conditions, the following parameters are varied:

| Parameter                  | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| Dataset                    | Determines the task type (image vs tabular classification)     |
| Client Participation Ratio | Fraction of clients participating in each communication round  |
| Local Epochs               | Number of training epochs performed locally before aggregation |

These parameters are evaluated across a full **experiment grid**, producing **36 independent experiments**.

---

## 3. Experiment Automation

The pipeline automates the full experimental workflow:

1. Generate configuration files dynamically
2. Execute the experiment using the **Fluke CLI**
3. Collect performance metrics
4. Store experiment outputs
5. Aggregate results across runs

Each experiment runs independently and produces its own output directory containing metrics and logs.

---

## 4. Performance Measurement

Model performance is evaluated using **classification accuracy** measured on the global test dataset.

For each run, the pipeline extracts:

```
global_metrics.csv
```

The **final round accuracy** is used as the main evaluation metric.

This value represents the final global model performance after federated aggregation.

---

## 5. Result Analysis

After all experiments complete, the pipeline aggregates results to enable comparison across:

* Training modes (centralized vs federated)
* Datasets
* Client participation ratios
* Local training epochs

Visualization tools such as **Matplotlib** and **TensorBoard** are used to analyze trends in training performance.

---

## 6. Reproducibility

To ensure reproducibility:

* All experiment configurations are generated programmatically.
* The same pipeline executes every experiment.
* All outputs are stored in versioned run directories.

This guarantees that results can be reproduced by rerunning the notebook or pipeline with the same configuration.

---

# Experiment Design

The experiment grid contains the following parameters:

| Parameter    | Values                  |
| ------------ | ----------------------- |
| Mode         | centralized, federation |
| Dataset      | mnist, adult            |
| Client Ratio | 0.3, 0.6, 1.0           |
| Local Epochs | 5, 10, 20               |

Total experiments:

2 × 2 × 3 × 3 = **36 experiments**

---

# Datasets

## MNIST

* Image classification dataset
* Model used: **MNIST_2NN**
* Loss: CrossEntropyLoss

## Adult

* Tabular income classification dataset
* Model used: **Adult_LogReg**
* Input dimension: **14 features**

Dataset-specific models are automatically selected by the pipeline.

---

# Pipeline Structure

The pipeline is composed of **10 structured steps**.

## Step 1 — Environment Setup

Installs required dependencies:

* PyTorch
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* PyYAML
* TensorBoard

---

## Step 2 — Repository Setup

Clones the FL-Lab repository and installs the **Fluke framework**.

---

## Step 3 — Experiment Grid Definition

Defines the parameter combinations that generate **36 experiments**.

---

## Step 4 — Dynamic Configuration Generation

Automatically updates:

* `exp.yaml`
* `fedavg.yaml`

for each experiment.

This step ensures correct configuration for different datasets.

Example:

* MNIST → `MNIST_2NN`
* Adult → `Adult_LogReg` with `input_dim = 14`

---

## Step 5 — Experiment Execution

Runs experiments via the **Fluke CLI**:

```
fluke <mode> config/tmp_exp.yaml config/tmp_alg.yaml
```

Safe execution ensures the pipeline continues even if an individual experiment fails.

---

## Step 6 — Sequential Experiment Loop

Executes all **36 experiments automatically**.

For each experiment:

1. Update configuration
2. Run Fluke
3. Collect metrics

---

## Step 7 — Metrics Collection

After each run, the pipeline extracts:

```
runs/run_X/global_metrics.csv
```

Final model accuracy is recorded.

---

## Step 8 — Experiment Output

Displays experiment configuration and results:

* Mode
* Dataset
* Client ratio
* Local epochs
* Model parameters
* Final accuracy

---

## Step 9 — Result Aggregation

Creates a summary table showing the **best accuracy per dataset and mode**.

Example:

| Mode        | Dataset | Max Accuracy |
| ----------- | ------- | ------------ |
| centralized | mnist   | ...          |
| centralized | adult   | ...          |
| federation  | mnist   | ...          |
| federation  | adult   | ...          |

---

## Step 10 — Visualization

The pipeline generates:

* **TensorBoard logs**
* **Accuracy comparison plots**

Example visualization:

```
Accuracy vs Client Ratio
```

Grouped by:

* Local epochs
* Training mode

---

# Output Artifacts

The pipeline generates:

```
runs/
 ├── run_1/
 ├── run_2/
 ├── ...
 ├── run_36/
```

Each run contains:

```
global_metrics.csv
TensorBoard logs
training statistics
```

---

# Running the Pipeline

The pipeline is designed for **Google Colab**.

Steps:

1. Open the notebook
2. Run all cells
3. Wait for the 36 experiments to finish
4. View results and plots

TensorBoard runs on:

```
localhost:6006
```

---

# Example Research Questions

This pipeline allows investigation of:

* Does **federated learning match centralized performance**?
* How does **client participation ratio** affect accuracy?
* Does increasing **local training epochs** improve results?
* How do results differ between **image vs tabular datasets**?

---

# Future Improvements

Possible extensions include:

* Additional FL algorithms (FedProx, SCAFFOLD, FedOpt)
* More datasets
* Communication efficiency analysis
* Client heterogeneity experiments
* Privacy-preserving FL techniques

---
