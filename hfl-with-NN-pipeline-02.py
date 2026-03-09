# ===============================
# Pipeline 2: Small PyTorch NN (Final)
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fluke.data.medical import Medical
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd

# --- Load processed dataset ---
processed_csv = "/content/fl-lab/data/medical_processed.csv"
medical_dataset = Medical(path=processed_csv)
X_train, y_train = medical_dataset.container.train
X_test, y_test = medical_dataset.container.test

num_clients = 5
num_rounds = 3
batch_size = 64
lr = 0.01

# --- Define small NN ---
class SmallNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- Client splitting functions ---
def split_iid(X, y, num_clients):
    samples_per_client = X.shape[0] // num_clients
    client_data = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = (i+1) * samples_per_client
        client_data.append((X[start:end], y[start:end]))
    return client_data

def split_noniid(X, y, num_clients, alpha=0.5):
    # Dirichlet-based non-IID split
    y_np = y.numpy()
    idx_by_class = [np.where(y_np == c)[0] for c in np.unique(y_np)]
    client_idx = [[] for _ in range(num_clients)]
    for class_idx in idx_by_class:
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet([alpha]*num_clients)
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)
        prev = 0
        for i, p in enumerate(proportions):
            client_idx[i].extend(class_idx[prev:p])
            prev = p
    client_data = [(X[ids], y[ids]) for ids in client_idx]
    return client_data

# --- Federated training function ---
def run_federated(X_train, y_train, client_split_fn, split_name):
    print(f"\n--- Running {split_name} Experiment ---")
    
    # Split clients
    client_data = client_split_fn(X_train, y_train)
    for i, (Xc, yc) in enumerate(client_data):
        print(f"Client {i+1}: {Xc.shape[0]} samples")
    
    # Initialize global model
    input_dim = X_train.shape[1]
    global_model = SmallNN(input_dim)
    criterion = nn.CrossEntropyLoss()
    
    # Store client-level metrics
    client_metrics = {f"Client {i+1}": [] for i in range(len(client_data))}
    
    # Global rounds
    for rnd in range(1, num_rounds+1):
        print(f"\n=== Global Round {rnd} ===")
        global_state = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        
        for idx, (Xc, yc) in enumerate(client_data):
            local_model = SmallNN(input_dim)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=lr)
            loader = DataLoader(TensorDataset(Xc, yc), batch_size=batch_size, shuffle=True)
            
            # Local training 1 epoch
            local_model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                out = local_model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            
            # Accumulate parameters for FedAvg
            for k in global_state:
                global_state[k] += local_model.state_dict()[k] / num_clients
            
            # Local evaluation
            local_model.eval()
            with torch.no_grad():
                y_pred_local = torch.argmax(local_model(Xc), dim=1)
                acc_local = (y_pred_local == yc).float().mean().item()
                client_metrics[f"Client {idx+1}"].append(acc_local)
            print(f"Client {idx+1} local accuracy: {acc_local:.4f}")
        
        # Update global model
        global_model.load_state_dict(global_state)
        
        # Global evaluation on full training set
        global_model.eval()
        with torch.no_grad():
            y_pred_global = torch.argmax(global_model(X_train), dim=1)
            acc_global = (y_pred_global == y_train).float().mean().item()
        print(f"[Global Round {rnd}] Global Accuracy: {acc_global:.4f}")
    
    # Final test metrics
    global_model.eval()
    with torch.no_grad():
        y_pred_test = torch.argmax(global_model(X_test), dim=1)
    acc_test = accuracy_score(y_test.numpy(), y_pred_test.numpy())
    f1_test = f1_score(y_test.numpy(), y_pred_test.numpy())
    prec_test = precision_score(y_test.numpy(), y_pred_test.numpy())
    rec_test = recall_score(y_test.numpy(), y_pred_test.numpy())
    
    # Fairness metrics (SPD & EOD by Age)
    age = pd.read_csv(processed_csv)["Age"].iloc[-X_test.shape[0]:].values
    age_bin = (age >= 50).astype(int)
    p_y1_age0 = np.mean(y_pred_test[age_bin == 0].numpy())
    p_y1_age1 = np.mean(y_pred_test[age_bin == 1].numpy())
    spd_age = p_y1_age0 - p_y1_age1
    eod_age = (p_y1_age0 - np.mean(y_test[age_bin == 0].numpy())) - (p_y1_age1 - np.mean(y_test[age_bin == 1].numpy()))
    
    # Print summary tables
    print("\n=== Performance Metrics ===")
    print("Metric   === Value")
    print(f"Accuracy === {acc_test:.4f}")
    print(f"F1 Score === {f1_test:.4f}")
    print(f"Precision === {prec_test:.4f}")
    print(f"Recall === {rec_test:.4f}")
    
    print("\n=== Fairness Metrics (Age) ===")
    print("Metric === Value")
    print(f"SPD (Age) === {spd_age:.4f}")
    print(f"EOD (Age) === {eod_age:.4f}")
    
    print("\n=== Utility Metrics ===")
    print("Metric === Value")
    print(f"Accuracy === {acc_test:.4f}")
    print(f"F1 Score === {f1_test:.4f}")
    
    print("\n=== Cost Metrics ===")
    print("Metric === Value")
    print(f"Global Rounds === {num_rounds}")
    print(f"Number of Clients === {len(client_data)}")
    
    return global_model, acc_test, f1_test, prec_test, rec_test, client_metrics

# --- Run experiments ---
global_model_iid, acc_iid, f1_iid, prec_iid, rec_iid, client_metrics_iid = run_federated(
    X_train, y_train, 
    lambda X, y: split_iid(X, y, num_clients),
    "IID"
)

global_model_noniid, acc_noniid, f1_noniid, prec_noniid, rec_noniid, client_metrics_noniid = run_federated(
    X_train, y_train, 
    lambda X, y: split_noniid(X, y, num_clients, alpha=0.5),
    "Non-IID"
)

# --- Client-level comparison (optional print) ---
print("\n=== Client-level Accuracy Comparison (IID vs Non-IID) ===")
print("Client === IID === Non-IID")
for i in range(num_clients):
    print(f"{i+1} === {client_metrics_iid[f'Client {i+1}'][-1]:.4f} === {client_metrics_noniid[f'Client {i+1}'][-1]:.4f}")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Prepare data for plots
rounds = [1, 2, 3]
clients = [f"Client {i+1}" for i in range(num_clients)]

# Global accuracy per round (approximate from last training rounds)
# If you want, you can store during run_federated and use exact per-round global accuracies
global_acc_iid = [0.6665, 0.6669, 0.6669]
global_acc_noniid = [0.6657, 0.6666, 0.6664]

# Client-level final accuracies
acc_iid_clients = [client_metrics_iid[f"C{i+1}"][-1] if f"C{i+1}" in client_metrics_iid else client_metrics_iid[f"Client {i+1}"][-1] for i in range(num_clients)]
acc_noniid_clients = [client_metrics_noniid[f"C{i+1}"][-1] if f"C{i+1}" in client_metrics_noniid else client_metrics_noniid[f"Client {i+1}"][-1] for i in range(num_clients)]

import matplotlib.pyplot as plt
import numpy as np

# --- Data from your latest results ---
num_rounds = 3
num_clients = 5
clients = [f"Client {i+1}" for i in range(num_clients)]

# Global accuracy per round
global_acc_iid = [0.6659, 0.6670, 0.6669]
global_acc_noniid = [0.6659, 0.6669, 0.6659]

# Client-level accuracy (final round)
client_acc_iid = [0.6685, 0.6661, 0.6684, 0.6646, 0.6662]
client_acc_noniid = [0.5621, 0.9531, 0.8766, 0.7460, 0.4157]

# Sample sizes per client
samples_iid = [8880]*num_clients
samples_noniid = [26225, 1130, 14453, 126, 2466]

# Fairness metrics (Age)
fairness_metrics = ['SPD (Age)', 'EOD (Age)']
spd_iid, eod_iid = 0.0000, -0.0029
spd_noniid, eod_noniid = -0.0008, -0.0037

# --- Plot setup ---
fig, axes = plt.subplots(2, 2, figsize=(15,10))
width = 0.35

# 1. Global Accuracy per Round
x_rounds = np.arange(num_rounds)
axes[0,0].plot(x_rounds, global_acc_iid, marker='o', label='IID')
axes[0,0].plot(x_rounds, global_acc_noniid, marker='s', label='Non-IID')
axes[0,0].set_title("Global Accuracy per Round")
axes[0,0].set_xlabel("Round")
axes[0,0].set_ylabel("Accuracy")
axes[0,0].set_xticks(x_rounds)
axes[0,0].grid(True)
axes[0,0].legend()

# 2. Client-level Accuracy Comparison
x_clients = np.arange(num_clients)
axes[0,1].bar(x_clients - width/2, client_acc_iid, width, label='IID')
axes[0,1].bar(x_clients + width/2, client_acc_noniid, width, label='Non-IID')
axes[0,1].set_title("Client-level Accuracy")
axes[0,1].set_xticks(x_clients)
axes[0,1].set_xticklabels(clients)
axes[0,1].set_ylabel("Accuracy")
axes[0,1].grid(axis='y')
axes[0,1].legend()

# 3. Client Sample Sizes
axes[1,0].bar(x_clients - width/2, samples_iid, width, label='IID')
axes[1,0].bar(x_clients + width/2, samples_noniid, width, label='Non-IID')
axes[1,0].set_title("Client Sample Sizes")
axes[1,0].set_xticks(x_clients)
axes[1,0].set_xticklabels(clients)
axes[1,0].set_ylabel("Number of Samples")
axes[1,0].grid(axis='y')
axes[1,0].legend()

# 4. Fairness Metrics (Age)
x_fairness = np.arange(len(fairness_metrics))
axes[1,1].bar(x_fairness - width/2, [spd_iid, eod_iid], width, label='IID')
axes[1,1].bar(x_fairness + width/2, [spd_noniid, eod_noniid], width, label='Non-IID')
axes[1,1].set_title("Fairness Metrics (Age)")
axes[1,1].set_xticks(x_fairness)
axes[1,1].set_xticklabels(fairness_metrics)
axes[1,1].set_ylabel("Value")
axes[1,1].grid(axis='y')
axes[1,1].legend()

plt.tight_layout()
plt.show()
