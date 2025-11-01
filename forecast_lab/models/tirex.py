import numpy as np
import pandas as pd
from ..registry import register_model

SCHEMA = {"use_adam": {"type": "bool", "default": False, "label": "Pre-filter with AdaM"}}

@register_model("tirex", schema=SCHEMA)
def run_tirex(df: pd.DataFrame, cfg: dict):
    target = cfg["target"]
    y = pd.to_numeric(df[target], errors="coerce").dropna().to_numpy(float)
    lookback = int(cfg.get("lookback", 96))
    horizon  = int(cfg.get("horizon", 24))
    last = float(y[-1]) if len(y) else 0.0
    yhat = np.full(horizon, last)
    return {"model": "tirex", "yhat": yhat, "meta": {"lookback": lookback, "horizon": horizon}}
