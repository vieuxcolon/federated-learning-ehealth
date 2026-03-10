---

# Vertical Federated Learning (VFL) with FLuKE

This repository implements a **Vertical Federated Learning (VFL) pipeline** using a Kaggle healthcare dataset. The pipeline leverages the **FLuKE framework** for federated experiments, including client-server models, training, evaluation, and fairness analysis.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Key Details](#dataset-key-details)
3. [Data Preprocessing](#data-preprocessing)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Experiment Setup](#experiment-setup)
6. [Metrics and Evaluation](#metrics-and-evaluation)
7. [Usage](#usage)

---

## Project Overview

This project demonstrates vertical federated learning, where **features of the same dataset are split across multiple clients**, and the server aggregates activations to perform prediction.
Key highlights:

* Client-side **Small Neural Networks (SmallNN)** for local feature processing
* Server-side fully connected NN for **global aggregation**
* Evaluation on **accuracy, F1, precision, recall**, and **fairness metrics** (SPD, EOD for age)
* Experimentation with different **client splits**

---

## Dataset Key Details

The dataset is a **Kaggle healthcare dataset** (`prasad22/healthcare-dataset`) containing patient records with demographics, hospital info, medical tests, and outcomes.

### Raw Dataset (Before Preprocessing)

* **Shape:** 55,500 rows × 15 columns
* **Columns:**
  `['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition', 'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider', 'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date', 'Medication', 'Test Results']`
* **Missing Values:** None
* **Preview:**

| Name          | Age | Gender | Blood Type | Medical Condition | … | Test Results |
| ------------- | --- | ------ | ---------- | ----------------- | - | ------------ |
| Bobby JacksOn | 30  | Male   | B-         | Cancer            | … | Normal       |
| LesLie TErRy  | 62  | Male   | A+         | Obesity           | … | Inconclusive |
| DaNnY sMitH   | 76  | Female | A-         | Obesity           | … | Normal       |
| andrEw waTtS  | 28  | Female | O+         | Diabetes          | … | Abnormal     |
| adrIENNE bEll | 43  | Female | AB+        | Cancer            | … | Abnormal     |

**Retrieve raw dataset info (Colab / single cell):**

```python
import pandas as pd
df_raw = pd.read_csv("/content/fl-lab/data/healthcare_dataset.csv")
print(df_raw.shape, df_raw.columns.tolist())
print(df_raw.isna().sum())
print(df_raw.head())
```

### Preprocessed Dataset (After Feature Selection & Label Binarization)

* **Shape:** 55,500 rows × 5 columns
* **Columns:** `['Age', 'Billing Amount', 'Room Number', 'length_of_stay', 'label_binary']`
* **Missing Values:** None
* **Preview:**

| Age | Billing Amount | Room Number | length_of_stay | label_binary |
| --- | -------------- | ----------- | -------------- | ------------ |
| 30  | 18856.28       | 328         | 2              | 0            |
| 62  | 33643.33       | 265         | 6              | 1            |
| 76  | 27955.10       | 205         | 15             | 0            |
| 28  | 37909.78       | 450         | 30             | 1            |
| 43  | 14238.32       | 458         | 20             | 1            |

**Retrieve preprocessed dataset info (Colab / single cell):**

```python
import pandas as pd
df_processed = pd.read_csv("/content/fl-lab/data/medical_processed.csv")
print(df_processed.shape, df_processed.columns.tolist())
print(df_processed.isna().sum())
print(df_processed.head())
```

---

## Data Preprocessing

1. **Label Binarization:**
   `Test Results` → `label_binary` (0 = Normal, 1 = Abnormal/Inconclusive)

2. **Feature Selection:**
   Numeric columns: `['Age', 'Billing Amount', 'Room Number']`
   Optional: `length_of_stay = Discharge Date - Date of Admission`

3. **Missing Values:**
   All numeric columns filled with 0

4. **Saved Preprocessed CSV:**
   `/content/fl-lab/data/medical_processed.csv`

---

## Pipeline Architecture

* **Clients:** Each client holds a subset of features

  * SmallNN: Linear → ReLU
* **Server:** Aggregates concatenated client activations

  * Linear layer for classification (2 output classes)
* **Training:** Global rounds (default 3), batch size 64, SGD optimizer
* **Evaluation:**

  * Global accuracy and client-level accuracy per round
  * Fairness metrics based on Age (SPD, EOD)

---

## Experiment Setup

| Parameter         | Options / Values               |
| ----------------- | ------------------------------ |
| Number of Clients | 5                              |
| Global Rounds     | 3                              |
| Batch Size        | 64                             |
| Client Hidden Dim | 16                             |
| Server Output Dim | 2                              |
| Loss Function     | CrossEntropyLoss               |
| Optimizer         | SGD (lr=0.01)                  |
| Client Split      | IID, Non-IID (Dirichlet α=0.5) |

* **Note:** In vertical federated learning, splitting the dataset IID vs Non-IID is not strictly necessary since each client holds **different features**, which is inherently non-IID.

---

## Metrics and Evaluation

* **Performance Metrics:** Accuracy, F1 Score, Precision, Recall
* **Fairness Metrics:** SPD and EOD (age-based)
* **Utility Metrics:** Global accuracy over rounds
* **Cost Metrics:** Number of clients, global rounds

**Visualization:** Multi-panel plots for:

1. Global Accuracy per Round
2. Client-level Accuracy Comparison
3. Client Sample Sizes
4. Fairness Metrics

---

## Usage

1. Clone repo:

```bash
git clone https://git.liris.cnrs.fr/nbenarba/fl-lab.git
```

2. Install dependencies:

```bash
pip install torch pandas scikit-learn matplotlib kagglehub
```

3. Run the pipeline in Colab / local environment:

```python
# Load datasets
# Preprocess & save medical_processed.csv
# Run VFL training and evaluation
```

4. Visualize results:

* Global accuracy, client-level accuracy, sample sizes, fairness metrics

---

This README now **covers dataset raw & preprocessed info, preprocessing steps, pipeline structure, experiments, and evaluation metrics**.

---
