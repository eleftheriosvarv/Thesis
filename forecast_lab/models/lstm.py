"""
LSTM skeleton (80/20 train-test) — fill later.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np

def get_schema() -> Dict[str, Any]:
    return {
        "required_opts": [],
        "optional_opts": ["hidden_size", "epochs", "lr"],
        "description": "LSTM 80/20 split (skeleton)."
    }

def _sliding_windows(x: np.ndarray, lookback: int):
    X, Y = [], []
    for i in range(len(x) - lookback):
        X.append(x[i:i+lookback])
        Y.append(x[i+lookback])
    return np.array(X), np.array(Y)

def run(data: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    target = cfg["target"]
    lookback = int(cfg.get("lookback", 96))
    y = pd.to_numeric(data[target], errors="coerce").dropna().values.astype(np.float32)

    n = len(y)
    k = max(lookback+1, int(n * 0.8))
    y_train, y_test = y[:k], y[k:]
    Xtr, Ytr = _sliding_windows(y_train, lookback)
    Xte, Yte = _sliding_windows(np.concatenate([y_train[-lookback:], y_test]), lookback)

    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError("PyTorch not installed. Later on cloud: pip install torch torchvision torchaudio") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class TinyLSTM(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            h,_ = self.lstm(x)
            return self.fc(h[:,-1,:])

    hidden = int((cfg.get("opts") or {}).get("hidden_size", 32))
    epochs = int((cfg.get("opts") or {}).get("epochs", 5))
    lr = float((cfg.get("opts") or {}).get("lr", 1e-3))

    model = TinyLSTM(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr[...,None], dtype=torch.float32).to(device)
    Ytr_t = torch.tensor(Ytr[...,None], dtype=torch.float32).to(device)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = loss_fn(pred, Ytr_t)
        loss.backward()
        opt.step()

    Xte_t = torch.tensor(Xte[...,None], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        yhat = model(Xte_t).cpu().numpy().ravel().tolist()

    return {
        "model": "lstm",
        "yhat": yhat,
        "meta": {
            "train_len": len(y_train),
            "test_len": len(y_test),
            "lookback": lookback,
            "epochs": epochs,
            "hidden_size": hidden,
            "note": "Skeleton ready for cloud execution"
        }
    }
