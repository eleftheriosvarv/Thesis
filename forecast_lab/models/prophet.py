"""
Prophet skeleton (80/20 train-test) — fill later.
"""
from typing import Dict, Any
import pandas as pd

def get_schema() -> Dict[str, Any]:
    return {
        "required_opts": [],
        "optional_opts": ["timestamp_col"],
        "description": "Prophet 80/20 split (skeleton)."
    }

def _split_train_test(df: pd.DataFrame, ratio: float = 0.8):
    n = len(df)
    k = max(1, int(n * ratio))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def run(data: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    target = cfg["target"]
    ts_col = (cfg.get("opts") or {}).get("timestamp_col", None)

    y = pd.to_numeric(data[target], errors="coerce")
    if ts_col and ts_col in data.columns:
        ds = pd.to_datetime(data[ts_col], errors="coerce")
    else:
        ds = pd.RangeIndex(start=0, stop=len(y), step=1)
    df = pd.DataFrame({"ds": ds, "y": y}).dropna()

    train_df, test_df = _split_train_test(df, 0.8)

    try:
        from prophet import Prophet
    except Exception as e:
        raise RuntimeError("Prophet not installed. Later on cloud: pip install prophet cmdstanpy") from e

    m = Prophet()
    m.fit(train_df)
    fcst = m.predict(test_df[["ds"]])

    y_true = test_df["y"].to_list()
    y_pred = fcst["yhat"].to_list()

    return {
        "model": "prophet",
        "yhat": y_pred,
        "meta": {
            "train_len": len(train_df),
            "test_len": len(test_df),
            "note": "Skeleton ready for cloud execution"
        }
    }
