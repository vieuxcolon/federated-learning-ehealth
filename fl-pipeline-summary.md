# Federated Learning Mini Research Pipeline

## Overview

This repository contains a **Python-based experiment pipeline for running and analyzing Federated Learning (FL) experiments automatically** using the Fluke framework.

The pipeline orchestrates the full research workflow:

1. Environment setup
2. Framework installation
3. Experiment configuration generation
4. Automated experiment execution
5. Metrics collection
6. Results aggregation
7. Visualization and experiment monitoring

The goal is to **systematically evaluate Federated Learning configurations** across multiple datasets and hyperparameter settings.

---

# Pipeline Workflow

## 1. Environment Setup

The pipeline begins by verifying the Python environment and working directory.

It ensures the runtime environment is ready and checks that required command-line tools are available before running experiments.

---

## 2. Repository Setup and Installation

The pipeline automatically clones the FL research repository:

```
https://git.liris.cnrs.fr/nbenarba/fl-lab.git
```

It then installs the Fluke package in editable mode:

```
pip install -e fluke_package
```

This provides the `fluke` command line interface used to execute experiments.

---

## 3. Experiment Grid Definition

The script defines a grid of experiment configurations using combinations of:

| Parameter    | Description                                        |
| ------------ | -------------------------------------------------- |
| Mode         | Centralized training or federated learning         |
| Client Ratio | Percentage of clients participating per round      |
| Local Epochs | Number of training epochs performed on each client |
| Dataset      | Dataset used for training                          |

### Experiment Grid

```
mode: ["centralized", "federation"]
client_ratio: [0.3, 0.6, 1.0]
local_epochs: [5, 10, 20]
dataset: ["mnist", "adult"]
```

Datasets used in the experiments:

* MNIST
* Adult dataset

The Cartesian product of these parameters results in:

**36 total experiments.**

---

# Configuration Generation

For each experiment run, the pipeline dynamically generates temporary configuration files:

### Experiment Configuration

Parameters modified include:

* Client participation percentage
* Dataset selection
* Number of global training rounds
* Logging directory

The pipeline enforces:

```
n_rounds = 20
```

for each experiment.

### Algorithm Configuration

The algorithm configuration controls:

* Local training epochs
* Model architecture
* Model input dimensions

Models used:

| Dataset | Model                  |
| ------- | ---------------------- |
| MNIST   | 2-layer neural network |
| Adult   | Logistic regression    |

---

# Experiment Execution

Each experiment is executed through the Fluke CLI:

```
fluke <mode> <experiment_config.yaml> <algorithm_config.yaml>
```

Example:

```
fluke federation tmp_exp.yaml tmp_alg.yaml
```

In federated mode, the system simulates the standard FL workflow based on
Federated Averaging:

1. Server selects participating clients
2. Clients train locally on their data
3. Clients send model updates
4. Server aggregates updates
5. Process repeats for multiple rounds

---

# Runtime Monitoring

During execution the pipeline records:

* Total experiment runtime
* Per-round training duration
* Final model accuracy

Round durations are estimated by parsing the Fluke training output.

---

# Result Collection

After each experiment completes, the pipeline reads metrics from:

```
runs/run_<id>/global_metrics.csv
```

or

```
runs/run_<id>/metrics.csv
```

The following information is extracted:

* Run ID
* Training mode
* Dataset
* Client participation ratio
* Local training epochs
* Final accuracy
* Total runtime
* Per-round durations
* Model configuration

Each experiment result is stored as a structured record.

---

# Experiment Summary

All experiment results are aggregated into a pandas dataframe.

The pipeline computes summary statistics such as:

```
Maximum accuracy per training mode and dataset
```

Example summary table:

| Mode        | Dataset | Max Accuracy |
| ----------- | ------- | ------------ |
| centralized | mnist   | 0.98         |
| federation  | mnist   | 0.97         |
| centralized | adult   | 0.86         |
| federation  | adult   | 0.84         |

This enables quick comparison between centralized and federated training approaches.

---

# Visualization and Monitoring

The pipeline launches TensorBoard to visualize experiment logs.

TensorBoard is started using:

```
tensorboard --logdir runs/
```

Default access:

```
http://localhost:6006
```

Additionally, the script generates performance plots using **Matplotlib** and **Seaborn**.

Example visualization:

**Accuracy vs Client Participation Ratio**

The plot highlights the effect of:

* client participation rate
* number of local epochs
* training mode (centralized vs federated)

---

# Repository Output Structure

After execution, the repository structure will contain:

```
fl-lab/
│
├── runs/
│   ├── run_1/
│   │   └── global_metrics.csv
│   ├── run_2/
│   │   └── global_metrics.csv
│   └── ...
│
├── experiment_plots/
├── summary_tables/
└── logs/
```

Each `run_X` directory corresponds to one experiment configuration.

---

# Research Use Cases

This pipeline enables experimentation with several Federated Learning questions:

### Client Participation

Evaluate how partial client participation affects performance.

Example:

```
client_ratio = 0.3 vs 1.0
```

---

### Local Computation vs Communication

Study the effect of increasing client training effort:

```
local_epochs = 5, 10, 20
```

---

### Federated vs Centralized Training

Compare model accuracy and runtime between:

```
centralized training
federated learning
```

---

### Dataset Characteristics

Analyze FL behavior across different data types:

* image classification (MNIST)
* tabular data (Adult dataset)

---

# Summary

This repository provides an **automated Federated Learning research pipeline** that:

* Runs **36 experiment configurations**
* Compares **centralized and federated training**
* Collects **accuracy and runtime metrics**
* Generates **experiment summaries and visualizations**
* Supports monitoring through **TensorBoard**

The pipeline simplifies reproducible experimentation for evaluating Federated Learning systems and hyperparameter configurations.
