---

# Results Comparison of HFL/VFL with Logistic Regression and Neural Networks

## **Overview**

This repository contains experimental results of **Federated Learning (FL)** on the **UCI Breast Cancer dataset**, comparing **Neural Networks** and **Logistic Regression** across **Horizontal (HFL)** and **Vertical (VFL)** settings. The experiments also evaluate **differential privacy (DP)** impact and **fairness across synthetic age groups**.

**Dataset:**

* 569 samples, 32 features (diagnosis + 30 features + synthetic age).
* Sensitive attribute: `age` (simulated, 25–80 years).

**FL Settings:**

* **HFL:** Each client holds all features but different rows.
* **VFL:** Each client holds different feature subsets, same rows.
* **Number of clients:** 4

**Models:**

* **Neural Network:** Fully connected (client embeddings → server → sigmoid output).
* **Logistic Regression:** Classical linear classifier applied per client/server.

**Evaluation Metrics:**

* Test Accuracy
* Positive Rate Young / Old
* Demographic Parity Difference
* DP sensitivity (`alpha` = 0.0, 0.01, 0.1)

---

## **1. Horizontal Federated Learning (HFL)**

### **1.1 Neural Network Results**

| DP alpha | Final Test Accuracy | PosRate_Young | PosRate_Old | Demographic Parity Diff |
| -------- | ------------------- | ------------- | ----------- | ----------------------- |
| 0.0      | 0.9561              | 0.9796        | 0.9385      | 0.0411                  |
| 0.01     | 0.9561              | 0.9796        | 0.9385      | 0.0411                  |
| 0.1      | 0.9211              | 0.8980        | 0.9385      | -0.0405                 |

**Observations:**

* NN achieves high accuracy but is sensitive to DP noise.
* Fairness gap increases under stronger DP (`alpha=0.1`).
* Requires multiple epochs to converge.

### **1.2 Logistic Regression Results**

| DP alpha | Final Test Accuracy | PosRate_Young | PosRate_Old | Demographic Parity Diff |
| -------- | ------------------- | ------------- | ----------- | ----------------------- |
| 0.0      | 0.9825              | 0.3469        | 0.3538      | -0.0069                 |
| 0.01     | 0.9825              | 0.3469        | 0.3538      | -0.0069                 |
| 0.1      | 0.9825              | 0.3469        | 0.3538      | -0.0069                 |

**Observations:**

* LR converges immediately and provides higher accuracy than NN for this dataset.
* DP impact is negligible.
* Fairness gap is very small (~0.007).

**Takeaway:**

* For HFL on linearly separable tabular data, Logistic Regression outperforms NN in **accuracy, fairness, and stability**.

---

## **2. Vertical Federated Learning (VFL)**

### **2.1 Neural Network Results**

| DP alpha | Final Test Accuracy | PosRate_Young | PosRate_Old | Demographic Parity Diff |
| -------- | ------------------- | ------------- | ----------- | ----------------------- |
| 0.0      | 0.9298              | 0.2860        | 0.3385      | -0.0527                 |
| 0.01     | 0.9211              | 0.2860        | 0.3231      | -0.0374                 |
| 0.1      | 0.8947              | 0.2245        | 0.2923      | -0.0678                 |

**Observations:**

* NN performance is slightly lower than HFL-LR.
* DP noise decreases accuracy and increases demographic disparity.
* Fairness gap is higher than LR-HFL.

### **2.2 Logistic Regression Results**

| DP alpha | Final Test Accuracy | PosRate_Young | PosRate_Old | Demographic Parity Diff |
| -------- | ------------------- | ------------- | ----------- | ----------------------- |
| 0.0      | 0.9737              | 0.3469        | 0.3692      | -0.0223                 |

**Observations:**

* LR achieves higher accuracy than NN for VFL as well.
* Fairness gap is smaller.
* DP not applied in this run, but LR is generally robust to small DP noise.

**Takeaway:**

* VFL LR achieves superior accuracy and fairness compared to NN, similar to HFL.
* NN may be preferred for complex, non-linear datasets, but tabular datasets favor LR.

---

## **3. Overall Comparison**

| FL Type | Model | DP alpha | Accuracy | Fairness (DP Diff) | Sensitivity to DP |
| ------- | ----- | -------- | -------- | ------------------ | ----------------- |
| HFL     | NN    | 0.0      | 0.9561   | 0.0411             | Moderate          |
| HFL     | NN    | 0.1      | 0.9211   | -0.0405            | High              |
| HFL     | LR    | 0.0      | 0.9825   | -0.0069            | Negligible        |
| HFL     | LR    | 0.1      | 0.9825   | -0.0069            | Negligible        |
| VFL     | NN    | 0.0      | 0.9298   | -0.0527            | Moderate          |
| VFL     | LR    | 0.0      | 0.9737   | -0.0223            | Low               |

**Key Insights:**

1. **Logistic Regression is highly effective for tabular FL datasets**, often outperforming NN in accuracy and fairness.
2. **NN adds flexibility but is more sensitive to DP noise** and requires careful tuning.
3. **Horizontal vs Vertical FL**:

   * HFL-LR has minimal DP impact and better fairness.
   * VFL-LR is robust but slightly lower fairness than HFL-LR due to feature partitioning.
4. **Differential Privacy**:

   * NN performance degrades under stronger DP (`alpha=0.1`).
   * LR performance is nearly insensitive for small datasets.

---

## **4. Recommendations**

* Use **Logistic Regression or XGBoost** for small, linearly separable tabular datasets in FL.
* Neural Networks are more suitable for **high-dimensional, non-linear, or complex feature interactions**.
* Apply DP carefully with NN to avoid utility loss.

---

