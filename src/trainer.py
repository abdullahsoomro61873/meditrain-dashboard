
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Neural Network
class CancerClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=16, dropout=0.3):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)


# Training Function
def train(cfg, data, on_epoch=None):

    # Unpack training and validation data only
    X_train, X_val = data[0], data[1]
    y_train, y_val = data[3], data[4]

    # Configuration
    hidden_size = cfg.get("hidden_size", 16)
    dropout = cfg.get("dropout", 0.3)
    learning_rate = cfg.get("learning_rate", 0.001)
    weight_decay = cfg.get("weight_decay", 0.0)
    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 32)
    patience = cfg.get("patience", 20)
    early_stopping = cfg.get("early_stopping", True)
    min_delta = cfg.get("min_delta", 0.0)
    seed = cfg.get("seed", 42)

    # Reproducibility
    torch.manual_seed(seed)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    y_train = torch.tensor(
        y_train, dtype=torch.float32
    ).reshape(-1, 1)

    y_val = torch.tensor(
        y_val, dtype=torch.float32
    ).reshape(-1, 1)

    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Initialize model
    model = CancerClassifier(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        dropout=dropout
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Training history
    hist = {
        "train": [],
        "val": []
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # Epoch training loop
    for epoch in range(1, epochs + 1):

        model.train()
        train_loss = 0.0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        train_loss /= len(dataset)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            outputs = model(X_val)

            loss = criterion(outputs, y_val)
            val_loss = loss.item()

        # Save losses
        hist["train"].append(train_loss)
        hist["val"].append(val_loss)

        # Track best validation loss
        if val_loss < best_val_loss - min_delta:

            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        # Callback after every epoch
        if on_epoch is not None:
            on_epoch(epoch, hist)

        # Early stopping
        if (
            early_stopping
            and epochs_without_improvement >= patience
        ):
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Training information
    hist["best_val_loss"] = min(hist["val"])
    hist["epochs_run"] = len(hist["train"])

    return model, hist
