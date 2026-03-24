"""
Baseline deep learning models for fraud detection.

Implements DNN, LSTM, GRU, and CNN1D in PyTorch with a unified API.
These serve as comparison baselines against TabNet.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from config import DNN_PARAMS, LSTM_PARAMS, GRU_PARAMS, CNN1D_PARAMS, RANDOM_SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── DNN ─────────────────────────────────────────────────────────────────────

class DNNModel(nn.Module):
    """Deep Neural Network: 3 hidden layers (256→128→64) with BatchNorm and Dropout."""

    def __init__(self, input_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in DNN_PARAMS["hidden_layers"]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(DNN_PARAMS["dropout"]),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── LSTM ────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """LSTM model: treats each feature as a sequence step."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=LSTM_PARAMS["hidden_size"],
            num_layers=LSTM_PARAMS["num_layers"],
            batch_first=True,
            dropout=LSTM_PARAMS["dropout"] if LSTM_PARAMS["num_layers"] > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(LSTM_PARAMS["hidden_size"], 64),
            nn.ReLU(),
            nn.Dropout(LSTM_PARAMS["dropout"]),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Reshape: (batch, features) → (batch, features, 1) for sequence input
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        out = lstm_out[:, -1, :]
        return self.fc(out).squeeze(-1)


# ─── GRU ─────────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    """GRU model: similar to LSTM but with GRU cells."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=GRU_PARAMS["hidden_size"],
            num_layers=GRU_PARAMS["num_layers"],
            batch_first=True,
            dropout=GRU_PARAMS["dropout"] if GRU_PARAMS["num_layers"] > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(GRU_PARAMS["hidden_size"], 64),
            nn.ReLU(),
            nn.Dropout(GRU_PARAMS["dropout"]),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        return self.fc(out).squeeze(-1)


# ─── CNN1D ───────────────────────────────────────────────────────────────────

class CNN1DModel(nn.Module):
    """1D Convolutional Network: Conv1D(64) → MaxPool → Conv1D(128) → Dense."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, features)
            nn.Conv1d(1, CNN1D_PARAMS["filters"][0], kernel_size=CNN1D_PARAMS["kernel_size"], padding=1),
            nn.BatchNorm1d(CNN1D_PARAMS["filters"][0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(CNN1D_PARAMS["filters"][0], CNN1D_PARAMS["filters"][1], kernel_size=CNN1D_PARAMS["kernel_size"], padding=1),
            nn.BatchNorm1d(CNN1D_PARAMS["filters"][1]),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(CNN1D_PARAMS["filters"][1], 64),
            nn.ReLU(),
            nn.Dropout(CNN1D_PARAMS["dropout"]),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Reshape: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc(x).squeeze(-1)


# ─── Unified Training API ────────────────────────────────────────────────────

MODEL_CLASSES = {
    "DNN": (DNNModel, DNN_PARAMS),
    "LSTM": (LSTMModel, LSTM_PARAMS),
    "GRU": (GRUModel, GRU_PARAMS),
    "CNN1D": (CNN1DModel, CNN1D_PARAMS),
}


def train_baseline_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[nn.Module, list]:
    """
    Train a baseline PyTorch model with early stopping.

    Returns:
        Trained model and list of training history dicts.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model_class, params = MODEL_CLASSES[model_name]
    input_dim = X_train.shape[1]
    model = model_class(input_dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCELoss()

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False
    )

    # Training loop with early stopping
    best_auc = 0
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(params["max_epochs"]):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
                val_preds.extend(y_pred.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  [{model_name}] Epoch {epoch+1:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f}"
            )

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  [{model_name}] Best Val AUC: {best_auc:.4f}")
    return model, history


def predict_baseline_model(
    model: nn.Module, X: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    """Get prediction probabilities from a baseline model."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)
            preds.extend(y_pred.cpu().numpy())

    return np.array(preds)
