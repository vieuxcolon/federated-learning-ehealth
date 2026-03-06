---

# How Federated Learning Works

This document explains federated learning concepts, key parameters, metrics, and workflow in the context of the current pipeline. A worked example using the Adult dataset demonstrates client- and server-level calculations.

---

## 1. Number of Clients per Round

**What it means:**

* In federated learning, the server selects a subset of clients for each round to perform local training.
* This determines how much of the global data distribution is represented in each round.

**How it works:**

* Selected clients train locally and send model updates to the server.
* The server aggregates updates to form the new global model.

**Parameter influencing it:**

* `client_ratio` (Fluke YAML: `protocol.eligible_perc`) → fraction of total clients participating per round.

  * Example: `client_ratio = 0.3` → 30% of clients selected each round.

---

## 2. Data Sampling

**What it means:**

* Data sampling determines how each client’s local dataset is chosen from the global dataset.

**How it works:**

* Only selected clients’ local data is used for training per round.
* Clients can have IID or non-IID partitions.

**Parameter influencing it:**

* `client_ratio` → affects which clients’ data is included per round.
* Data partitioning logic during preprocessing controls IID vs non-IID behavior.

---

## 3. Sharing of Data Among Clients

**What it means:**

* Federated learning does **not share raw data**; only model updates are exchanged.

**How it works:**

* Clients compute gradients or updated model weights locally.
* Server aggregates updates (e.g., averaging in FedAvg) to update the global model.

**Parameter influencing it:**

* `mode="federation"` → enables federated learning mode.
* `client_ratio` → determines which clients send updates to server.

---

## 4. Round vs Epoch vs Local Runs

| Term                 | Definition                                                           |
| -------------------- | -------------------------------------------------------------------- |
| Round (Global Round) | One server aggregation step after receiving client updates           |
| Epoch (Local Epoch)  | One pass over a client’s local dataset                               |
| Local Runs           | Number of updates a client performs before sending updates to server |

**How it works:**

* Clients train for `local_epochs` per round.
* Server aggregates after each round.

**Parameters influencing it:**

* `protocol.n_rounds` → number of global rounds.
* `hyperparameters.client.local_epochs` → local epochs per client.

---

## 5. IID vs Non-IID Data

**What it means:**

* **IID:** Each client has a representative subset of the global distribution.
* **Non-IID:** Clients have skewed or biased subsets.

**How it works:**

* Non-IID data increases heterogeneity, affecting convergence and fairness.

**Parameter influencing it:**

* Data partitioning logic in preprocessing determines IID vs non-IID behavior.

---

## 6. Metrics

| Metric Type      | Measures                         | Parameter Influence                        |
| ---------------- | -------------------------------- | ------------------------------------------ |
| Utility Metrics  | Model performance (accuracy, F1) | Depends on dataset, model, rounds, epochs  |
| Quality Metrics  | Convergence, stability           | `protocol.n_rounds`, `local_epochs`        |
| Fairness Metrics | Equity among clients             | Non-IID data, client selection             |
| Cost Metrics     | Communication & computation      | `client_ratio`, model size, `local_epochs` |

---

## 7. Algorithms: Local vs Federated

* **Local Training:** Each client trains independently; no aggregation.

  * Parameter: `mode="centralized"`.

* **Federated Training:** Clients’ updates are aggregated by the server.

  * Parameters: `mode="federation"`, `client_ratio`, `local_epochs`, `protocol.n_rounds`.

---

## 8. Vertical vs Horizontal Federated Learning

| Aspect       | Horizontal FL                 | Vertical FL                                 |
| ------------ | ----------------------------- | ------------------------------------------- |
| Data Overlap | Different rows, same features | Same rows, different features               |
| Data Sharing | Only model updates            | Intermediate outputs or encrypted gradients |
| Use Cases    | Hospitals, mobile devices     | Banks, joint user profiling                 |

**Parameters influencing it:**

* HFL → `mode="federation"`, `client_ratio`, dataset split by rows
* VFL → `mode="federation"`, secure aggregation, feature alignment across clients

---

## 9. Difference Between Sharing Data and Sampling Data

| Concept       | What                                     | When it Happens           | Parameter                                          |
| ------------- | ---------------------------------------- | ------------------------- | -------------------------------------------------- |
| Sharing Data  | Clients send model updates, not raw data | During server aggregation | `mode="federation"`, aggregation method            |
| Sampling Data | Subset of clients is selected per round  | Beginning of each round   | `client_ratio`, total rounds (`protocol.n_rounds`) |

---

## 10. Example: Adult Dataset — 3 Rounds

**Configuration:**

| Parameter     | Value                                |
| ------------- | ------------------------------------ |
| Dataset       | Adult                                |
| Total Clients | 10                                   |
| Client Ratio  | 0.3 → 3 clients per round            |
| Local Epochs  | 5                                    |
| Global Rounds | 3                                    |
| Model         | Logistic Regression (`Adult_LogReg`) |

---

### Step 1: Round 1

* Server selects 3 clients randomly.
* Each client trains for 5 local epochs on its local Adult subset.
* Clients send model weights to server.
* Server aggregates via FedAvg:

```python
w_global = (w_C1 + w_C2 + w_C3)/3
```

---

### Step 2: Round 2

* Server selects next 3 clients (may overlap).
* Local training and aggregation.
* Global model updated.

---

### Step 3: Round 3

* Server selects next 3 clients.
* Local training → aggregation → final global model.

---

**Per-Round Calculations Example (Simplified):**

| Round | Clients Selected | Aggregation Formula                 | Example Accuracy |
| ----- | ---------------- | ----------------------------------- | ---------------- |
| 1     | C1, C2, C3       | `w_global = (w_C1 + w_C2 + w_C3)/3` | 0.9501           |
| 2     | C4, C5, C6       | `w_global = (w_C4 + w_C5 + w_C6)/3` | 0.9610           |
| 3     | C7, C8, C9       | `w_global = (w_C7 + w_C8 + w_C9)/3` | 0.9751           |

* Average per-round duration:

```python
avg_round_time_sec = total_duration / total_rounds
```

* Per-epoch durations tracked for profiling.

---

 This document now fully describes:

1. Key FL concepts and terminology
2. Horizontal vs vertical FL
3. Difference between client data sharing and sampling
4. All influencing parameters
5. Worked example with Adult dataset over 3 rounds

---

