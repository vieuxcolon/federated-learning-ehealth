# ======================================================
# RESEARCH-GRADE VERTICAL FEDERATED LEARNING PIPELINE
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
print("Why: Structured dataset required for VFL pipeline")
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
# Step 5: Vertical Feature Partitioning
# ======================================================
print("\nStep 5: Partition Features Across 4 Clients")
print("What: Simulate vertical federated learning")
print("Why: Each client owns feature subset")
print("How: Split feature columns evenly")

num_clients = 4
feature_splits = np.array_split(np.arange(X_train.shape[1]), num_clients)

def split_clients(X):
    return [X[:, idx] for idx in feature_splits]

X_train_clients = split_clients(X_train)
X_test_clients = split_clients(X_test)

# ======================================================
# Step 6: Model Factory (No Overwrites)
# ======================================================
print("\nStep 6: Model Factory Initialization")
print("What: Create fresh client/server models per experiment")
print("Why: Prevent weight leakage across DP runs")
print("How: Define factory function returning new models")

def init_models():
    client_models = []
    for i in range(num_clients):
        model = nn.Sequential(
            nn.Linear(len(feature_splits[i]), 8),
            nn.ReLU()
        ).to(device)
        client_models.append(model)

    server_model = nn.Sequential(
        nn.Linear(8 * num_clients, 1),
        nn.Sigmoid()
    ).to(device)

    return client_models, server_model

# ======================================================
# Step 7: VFL Core Functions
# ======================================================
print("\nStep 7: Define Training & Evaluation Logic")
print("What: Modular VFL forward, train, evaluate")
print("Why: Clean experiment isolation")
print("How: Client embeddings → concat → server")

def forward_vfl(X_clients, client_models, server_model):
    embeddings = []
    for i in range(num_clients):
        x = torch.tensor(X_clients[i], dtype=torch.float32).to(device)
        embeddings.append(client_models[i](x))
    concat = torch.cat(embeddings, dim=1)
    return server_model(concat)

def evaluate(X_clients, y, client_models, server_model):
    with torch.no_grad():
        outputs = forward_vfl(X_clients, client_models, server_model)
        preds = (outputs > 0.5).cpu().numpy().flatten()
        acc = (preds == y).mean()
    return acc, preds

def train_vfl(X_train_clients, y_train,
              X_test_clients, y_test,
              dp_alpha=0.0, epochs=20):

    client_models, server_model = init_models()

    criterion = nn.BCELoss()
    opt_server = optim.Adam(server_model.parameters(), lr=0.01)
    opt_clients = [optim.Adam(m.parameters(), lr=0.01) for m in client_models]

    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    history = {"train_acc":[], "test_acc":[]}

    for epoch in range(1, epochs+1):

        outputs = forward_vfl(X_train_clients, client_models, server_model)
        loss = criterion(outputs, y_tensor)

        opt_server.zero_grad()
        for opt in opt_clients:
            opt.zero_grad()

        loss.backward()

        # Gradient clipping
        for m in client_models + [server_model]:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)

        # DP Noise
        if dp_alpha > 0:
            for m in client_models + [server_model]:
                for p in m.parameters():
                    p.grad += dp_alpha * torch.randn_like(p.grad)

        opt_server.step()
        for opt in opt_clients:
            opt.step()

        train_acc, _ = evaluate(X_train_clients, y_train,
                                client_models, server_model)
        test_acc, _ = evaluate(X_test_clients, y_test,
                               client_models, server_model)

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch==1 or epoch%5==0 or epoch==epochs:
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    return client_models, server_model, history

# ======================================================
# Step 8: Differential Privacy Sweep
# ======================================================
print("\nStep 8: Differential Privacy Sweep")
print("What: Compare utility & fairness across DP levels")
print("Why: Study privacy–utility–fairness tradeoff")
print("How: Train fresh models per alpha and store histories")

dp_alphas = [0.0, 0.01, 0.1]

experiment_results = {}
summary_rows = []

for alpha in dp_alphas:

    print(f"\n--- Training with DP alpha = {alpha} ---")

    clients, server, history = train_vfl(
        X_train_clients, y_train,
        X_test_clients, y_test,
        dp_alpha=alpha
    )

    test_acc, test_preds = evaluate(
        X_test_clients, y_test,
        clients, server
    )

    # True Demographic Parity
    young = age_test <= 50
    old = age_test > 50

    pos_young = test_preds[young].mean()
    pos_old = test_preds[old].mean()
    dp_diff = pos_young - pos_old

    experiment_results[alpha] = {
        "clients": clients,
        "server": server,
        "history": history
    }

    summary_rows.append({
        "DP_alpha": alpha,
        "Final_Test_Accuracy": test_acc,
        "PosRate_Young": pos_young,
        "PosRate_Old": pos_old,
        "Demographic_Parity_Diff": dp_diff
    })

# ======================================================
# Step 9: Summary Table
# ======================================================
print("\nStep 9: Summary Table")

summary_df = pd.DataFrame(summary_rows)
print(summary_df)

# ======================================================
# Step 10: Plot Learning Curves (No Retraining)
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
plt.ylabel("Test Accuracy")
plt.title("VFL Test Accuracy vs Epoch")
plt.legend()
plt.show()
