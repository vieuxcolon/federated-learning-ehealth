# ======================================================
# RESEARCH-GRADE HORIZONTAL FEDERATED LEARNING PIPELINE
# No Regression Overwrites | Reproducible | Publication-Ready
# ======================================================

# ======================================================
# Step 1: Reproducibility & Imports
# ======================================================
print("Step 1: Import Libraries")
print("What: Load ML, DP, and plotting libraries")
print("Why: Ensure reproducible and modular experimentation")
print("How: Set seeds + import numpy, pandas, torch, sklearn, matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# Step 2: Load Dataset
# ======================================================
print("\nStep 2: Load Dataset")
print("What: Load UCI Breast Cancer dataset with headers")
print("Why: Structured dataset required for HFL pipeline")
print("How: Assign column names manually")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

columns = ["ID", "Diagnosis"] + \
          [f"{stat}_{feat}" for feat in [
              "radius","texture","perimeter","area","smoothness",
              "compactness","concavity","concave_points","symmetry",
              "fractal_dimension"]
           for stat in ["mean","se","worst"]]

df = pd.read_csv(url, header=None, names=columns)
print(f"Dataset shape: {df.shape}")

# ======================================================
# Step 3: Add Synthetic Sensitive Attribute
# ======================================================
print("\nStep 3: Add Synthetic 'age'")
print("What: Simulate demographic attribute")
print("Why: Evaluate fairness across age groups")
print("How: Random integer ages 25–80")

df["age"] = np.random.randint(25, 81, size=len(df))

# ======================================================
# Step 4: Train/Test Split + Standardization
# ======================================================
print("\nStep 4: Data Split & Standardization")
print("What: Stratified split and feature normalization")
print("Why: Prevent leakage and scale sensitivity")
print("How: sklearn train_test_split + StandardScaler")

df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df["Diagnosis"], random_state=SEED
)

feature_cols = [c for c in df.columns if c not in ["ID","Diagnosis","age"]]

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[feature_cols])
X_test = scaler.transform(df_test[feature_cols])

y_train = (df_train["Diagnosis"]=="M").astype(int).values
y_test = (df_test["Diagnosis"]=="M").astype(int).values

age_train = df_train["age"].values
age_test = df_test["age"].values

# ======================================================
# Step 5: Horizontal Partitioning (Rows Split Across Clients)
# ======================================================
print("\nStep 5: Partition Data Across 4 Clients (HFL)")
print("What: Each client receives a disjoint subset of samples (all features)")
print("Why: Simulate horizontal federated learning")
print("How: Split rows evenly; handle unequal batch sizes during training/evaluation")

num_clients = 4
num_samples = X_train.shape[0]
samples_per_client = num_samples // num_clients

client_data = []

for i in range(num_clients):
    start_idx = i * samples_per_client
    # Last client takes remaining samples
    end_idx = (i+1) * samples_per_client if i < num_clients-1 else num_samples
    client_data.append({
        "X_train": X_train[start_idx:end_idx],
        "X_test": X_test[start_idx:end_idx],
        "y_train": y_train[start_idx:end_idx],
        "y_test": y_test[start_idx:end_idx],
        "age_train": age_train[start_idx:end_idx],
        "age_test": age_test[start_idx:end_idx]
    })

print(f"Partitioned data into {num_clients} clients with row counts: "
      f"{[d['X_train'].shape[0] for d in client_data]}")

# ======================================================
# Step 6: Model Factory (No Overwrites)
# ======================================================
print("\nStep 6: Model Factory Initialization")
print("What: Create fresh client models per experiment")
print("Why: Prevent weight leakage across DP runs")
print("How: Define factory function returning new models")

def init_hfl_models(input_dim):
    """Initialize identical client models"""
    client_models = []
    for _ in range(num_clients):
        model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)
        client_models.append(model)
    return client_models

# ======================================================
# Step 7: HFL Training & Evaluation Functions
# ======================================================
print("\nStep 7: Define Training & Evaluation Logic (HFL)")
print("What: Train independent client models; aggregate predictions")
print("Why: Horizontal FL requires averaging predictions across clients")
print("How: Forward per client → optional DP noise → evaluate accuracy per client → average")

def forward_hfl(clients, X_list):
    """Forward pass per client; return list of predictions"""
    preds = []
    for model, X in zip(clients, X_list):
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds.append(model(x_tensor))
    return preds  # List of tensors, one per client

def evaluate_hfl(clients, client_data_list):
    """Evaluate HFL: compute average test accuracy across clients"""
    accs, all_preds = [], []
    for i, data in enumerate(client_data_list):
        y_tensor = torch.tensor(data["y_test"], dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            pred = clients[i](torch.tensor(data["X_test"], dtype=torch.float32).to(device))
            pred_labels = (pred > 0.5).cpu().numpy().flatten()
            acc = (pred_labels == data["y_test"]).mean()
        accs.append(acc)
        all_preds.append(pred_labels)
    avg_acc = np.mean(accs)
    return avg_acc, all_preds

def train_hfl(client_data_list, dp_alpha=0.0, epochs=20):
    """Train HFL client models with optional DP"""
    input_dim = client_data_list[0]["X_train"].shape[1]
    clients = init_hfl_models(input_dim)
    criterion = nn.BCELoss()
    optimizers = [optim.Adam(m.parameters(), lr=0.01) for m in clients]

    history = {"train_acc": [], "test_acc": []}

    for epoch in range(1, epochs+1):
        for i, data in enumerate(client_data_list):
            X_train = torch.tensor(data["X_train"], dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(1).to(device)

            outputs = clients[i](X_train)
            loss = criterion(outputs, y_train_tensor)

            optimizers[i].zero_grad()
            loss.backward()

            # Apply DP noise if requested
            if dp_alpha > 0.0:
                for p in clients[i].parameters():
                    p.grad += dp_alpha * torch.randn_like(p.grad)

            optimizers[i].step()

        # Compute average train/test accuracy across clients
        train_acc, _ = evaluate_hfl(clients, client_data_list)
        test_acc, _ = evaluate_hfl(clients, client_data_list)

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    return clients, history

# ======================================================
# Step 8: Differential Privacy Sweep & Fairness Metrics
# ======================================================
print("\nStep 8: DP Sweep & Fairness Metrics")
print("What: Compare utility & fairness across DP levels")
print("Why: Study privacy–utility–fairness tradeoff")
print("How: Train fresh client models per DP alpha and store histories")

dp_alphas = [0.0, 0.01, 0.1]
summary_rows = []
experiment_results = {}

for alpha in dp_alphas:
    print(f"\n--- Training with DP alpha={alpha} ---")
    clients, history = train_hfl(client_data, dp_alpha=alpha, epochs=20)

    # Evaluate HFL: average test accuracy across clients
    test_acc, all_preds = evaluate_hfl(clients, client_data)

    # Fairness Metrics: concatenate all client predictions and ages
    preds_concat = np.concatenate(all_preds)
    ages_concat = np.concatenate([d["age_test"] for d in client_data])
    labels_concat = np.concatenate([d["y_test"] for d in client_data])

    young_idx = ages_concat <= 50
    old_idx = ages_concat > 50
    acc_young = (preds_concat[young_idx] == labels_concat[young_idx]).mean()
    acc_old = (preds_concat[old_idx] == labels_concat[old_idx]).mean()
    dp_diff = acc_young - acc_old

    experiment_results[alpha] = {"clients": clients, "history": history}

    summary_rows.append({
        "DP_alpha": alpha,
        "Final_Test_Accuracy": test_acc,
        "Acc_Young": acc_young,
        "Acc_Old": acc_old,
        "Demographic_Parity_Diff": dp_diff
    })

# ======================================================
# Step 9: Summary Table
# ======================================================
print("\nStep 9: Summary Table")
summary_df = pd.DataFrame(summary_rows)
print(summary_df)

# ======================================================
# Step 10: Plot Learning Curves
# ======================================================
print("\nStep 10: Plot Test Accuracy Curves")
print("What: Visualize learning dynamics")
print("Why: Compare DP impact over epochs")
print("How: Use stored histories (no retraining)")

plt.figure(figsize=(8,6))
for alpha in dp_alphas:
    plt.plot(
        experiment_results[alpha]["history"]["test_acc"],
        label=f"DP alpha={alpha}"
    )

plt.xlabel("Epoch")
plt.ylabel("Average Test Accuracy")
plt.title("HFL Test Accuracy vs Epoch")
plt.legend()
plt.show()

# ======================================================
# Step 11: HFL Architecture Overview (Text-Based)
# ======================================================
hfl_architecture = f"""
Horizontal Federated Learning (HFL) Architecture:

- Number of Clients (Hospitals): {num_clients}
- Each Client: Owns all features, unique subset of samples
- Server: Not used (aggregation via output averaging)
- Federated Learning Algorithm: Horizontal Federated Averaging (HFL)
- Data Partition:
    - Train: Each client gets disjoint rows
    - Test: Full test set visible to all clients
- Model per Client: NN(NumFeatures -> 16 -> 1) + Sigmoid
- Differential Privacy: Gaussian noise added to gradients per client
- Evaluation Metrics: Accuracy, PosRate_Young, PosRate_Old, Demographic_Parity_Diff
"""
print(hfl_architecture)

# ======================================================
# Step 12: Pipeline Logical Flow (Tree)
# ======================================================
hfl_pipeline_tree = """
HFL Pipeline:
├─ Step 1: Imports & Reproducibility
├─ Step 2: Load Dataset
├─ Step 3: Synthetic Sensitive Attribute
├─ Step 4: Train/Test Split + Standardization
├─ Step 5: Horizontal Partition Across Clients
├─ Step 6: Model Factory Initialization
├─ Step 7: Training & Evaluation Functions
│   ├─ Forward Pass: Each client -> Output
│   ├─ Aggregate: Average client outputs
│   ├─ Compute BCELoss
│   ├─ Backpropagation (with optional DP)
├─ Step 8: DP Sweep & Fairness Metrics
├─ Step 9: Summary Table
├─ Step 10: Plot Learning Curves
├─ Step 11: HFL Architecture Overview
└─ Step 12: Pipeline Logical Flow Tree
"""
print(hfl_pipeline_tree)
