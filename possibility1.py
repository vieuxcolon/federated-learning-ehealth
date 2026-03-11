# ===========================================================================
# TRUE VFL RESEARCH PIPELINE (3 EXPERIMENTS, ADULT DATASET) - GOLDEN VERSION
# ===========================================================================

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Step 0 — Safeguard Functions
# -------------------------------

def safe_read_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return None


# -------------------------------
# Step 1 — Environment Setup
# -------------------------------

print("Step 1 — Environment Setup")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# -------------------------------
# Step 2 — Load Dataset (Adult)
# -------------------------------

print("Step 2 — Load Dataset")

repo_dir = os.path.join(os.getcwd(), "fl-lab")

dataset_path = os.path.join(repo_dir, "data", "adult", "adult.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Adult dataset not found at {dataset_path}"
    )

df = pd.read_csv(dataset_path)

# Convert categorical features
df = pd.get_dummies(df)

# Split features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("Dataset loaded successfully")
print("Training samples:", X_train.shape)
print("Test samples:", X_test.shape)


# -------------------------------
# Step 3 — Experiment Grid
# -------------------------------

print("Step 3 — Define Experiment Grid")

experiments_list = [
    {"local_epochs": 5},
    {"local_epochs": 10},
    {"local_epochs": 20}
]

print("Total experiments:", len(experiments_list))


# -------------------------------
# Step 4 — Feature Partition
# -------------------------------

print("Step 4 — Feature Partition")

num_clients = 3
total_features = X_train.shape[1]

split_size = total_features // num_clients

feature_splits = []

for i in range(num_clients):

    start = i * split_size

    if i < num_clients - 1:
        end = (i + 1) * split_size
    else:
        end = total_features

    feature_splits.append((start, end))

print("Feature splits:", feature_splits)


# -------------------------------
# Step 5 — Define VFL Models
# -------------------------------

class ClientEmbedding(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x):

        return self.net(x)


class ServerModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.classifier = nn.Linear(input_dim, 2)

    def forward(self, x):

        return self.classifier(x)


# -------------------------------
# Step 6 — VFL Training Function
# -------------------------------

def run_vfl(local_epochs):

    client_models = []
    client_optimizers = []

    for start, end in feature_splits:

        model = ClientEmbedding(end - start).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        client_models.append(model)
        client_optimizers.append(optimizer)

    server = ServerModel(8 * num_clients).to(device)

    server_optimizer = optim.Adam(server.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    rounds = 20

    start_time = time.time()

    for r in range(rounds):

        embeddings = []

        for i, (start, end) in enumerate(feature_splits):

            x_part = X_train[:, start:end].to(device)

            emb = client_models[i](x_part)

            embeddings.append(emb)

        concat = torch.cat(embeddings, dim=1)

        preds = server(concat)

        loss = criterion(preds, y_train.to(device))

        server_optimizer.zero_grad()

        for opt in client_optimizers:
            opt.zero_grad()

        loss.backward()

        server_optimizer.step()

        for opt in client_optimizers:
            opt.step()

    duration = round(time.time() - start_time, 3)

    return client_models, server, duration


# -------------------------------
# Step 7 — Run Experiments
# -------------------------------

results = []

run_id = 0

for exp in experiments_list:

    run_id += 1

    print("\nRunning VFL Experiment", run_id)

    client_models, server, duration = run_vfl(exp["local_epochs"])

    with torch.no_grad():

        embeddings = []

        for i, (start, end) in enumerate(feature_splits):

            x_part = X_test[:, start:end].to(device)

            emb = client_models[i](x_part)

            embeddings.append(emb)

        concat = torch.cat(embeddings, dim=1)

        preds = server(concat)

        y_pred = torch.argmax(preds, dim=1).cpu()

    acc = accuracy_score(y_test.numpy(), y_pred.numpy())

    results.append({

        "run_id": run_id,
        "local_epochs": exp["local_epochs"],
        "accuracy": acc,
        "duration": duration

    })


# -------------------------------
# Step 8 — Summary Table
# -------------------------------

results_df = pd.DataFrame(results)

print("\nResults")
print(results_df)


# -------------------------------
# Step 9 — Plot Results
# -------------------------------

plt.figure(figsize=(8, 5))

sns.lineplot(
    data=results_df,
    x="local_epochs",
    y="accuracy",
    marker="o"
)

plt.title("VFL Accuracy vs Local Epochs")

plt.show()

print("\nPipeline completed successfully.")
