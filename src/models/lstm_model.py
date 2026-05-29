"""LSTM model for sequence-based RUL prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMNet(nn.Module):
    """Simple LSTM network for RUL regression."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel:
    """Wrapper for training and inference with LSTM RUL model."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LSTMNet(input_size, hidden_size, num_layers).to(self.device)
        self.input_size = input_size
        self.sequence_length = 30

    def _build_sequences(
        self, df: pd.DataFrame, feature_cols: list[str], target_col: str | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        sequences, targets = [], []
        for _, group in df.groupby("unit_id"):
            values = group[feature_cols].values
            if target_col and target_col in group.columns:
                y_vals = group[target_col].values
            else:
                y_vals = None

            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i : i + self.sequence_length])
                if y_vals is not None:
                    targets.append(y_vals[i + self.sequence_length])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32) if targets else None
        return X, y

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "rul",
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> dict:
        """Train LSTM on sequential sensor data."""
        X, y = self._build_sequences(df, feature_cols, target_col)
        dataset = TensorDataset(
            torch.tensor(X), torch.tensor(y, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.net.train()
        losses = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.net(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        return {"final_loss": losses[-1], "loss_history": losses}

    def _sequence_tensor(self, values: np.ndarray, end_index: int) -> torch.Tensor:
        """Window ending at end_index (inclusive), left-padded if needed."""
        window = values[: end_index + 1]
        if len(window) < self.sequence_length:
            pad = np.repeat(window[:1], self.sequence_length - len(window), axis=0)
            seq = np.vstack([pad, window])
        else:
            seq = window[-self.sequence_length :]
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Generate RUL predictions from the latest sequence per unit (sorted by unit_id)."""
        self.net.eval()
        predictions = []
        with torch.no_grad():
            for _, group in df.groupby("unit_id", sort=True):
                values = group[feature_cols].values.astype(np.float32)
                if len(values) == 0:
                    predictions.append(0.0)
                    continue
                x = self._sequence_tensor(values, len(values) - 1)
                pred = float(self.net(x).cpu().item())
                predictions.append(max(0.0, pred))
        return np.array(predictions)

    def predict_validation(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict RUL at each non-terminal row (matches rul_validation_frame).

        Returns (y_true, y_pred) in unit_id / cycle order.
        """
        y_true: list[float] = []
        y_pred: list[float] = []
        self.net.eval()
        with torch.no_grad():
            full = df.sort_values(["unit_id", "cycle"])
            for _, group in full.groupby("unit_id", sort=True):
                values = group[feature_cols].values.astype(np.float32)
                y_vals = group["rul"].values
                cycles = group["cycle"].values
                terminal_cycle = cycles.max()
                for i in range(len(values)):
                    if cycles[i] >= terminal_cycle:
                        continue
                    x = self._sequence_tensor(values, i)
                    pred = max(0.0, float(self.net(x).cpu().item()))
                    y_true.append(float(y_vals[i]))
                    y_pred.append(pred)
        return np.array(y_true, dtype=float), np.array(y_pred, dtype=float)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "input_size": self.input_size,
                "sequence_length": self.sequence_length,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(input_size=checkpoint["input_size"])
        instance.net.load_state_dict(checkpoint["state_dict"])
        instance.sequence_length = checkpoint["sequence_length"]
        return instance
