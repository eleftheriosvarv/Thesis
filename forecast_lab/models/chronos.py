
import numpy as np
import pandas as pd
from ..registry import register_model

SCHEMA = {
    "use_adam": {"type": "bool", "default": False, "label": "Pre-filter with AdaM"},
    "few_shot": {"type": "bool", "default": False, "label": "Few-shot"},
    "shots":    {"type": "int",  "default": 4, "min": 0, "max": 64, "depends_on": "few_shot"},
    "prep_win": {"type": "int",  "default": 16, "label": "Prepared windows"},
}

@register_model("chronos", schema=SCHEMA)
def run_chronos(df: pd.DataFrame, cfg: dict):
    target = cfg["target"]
    y = pd.to_numeric(df[target], errors="coerce").dropna().to_numpy(float)
    lookback = int(cfg.get("lookback", 96))
    horizon  = int(cfg.get("horizon", 24))
    opts = cfg.get("opts", {})

    # Placeholder forward (damped trend). Replace with real Chronos inference.
    ctx = y[-min(lookback, len(y)) :]
    if len(ctx) < 2:
        slope = 0.0; base = ctx[-1]
    else:
        x = np.arange(len(ctx))
        slope = float(np.polyfit(x, ctx, 1)[0]) * 0.5
        base = ctx[-1]
    yhat = base + slope * np.arange(1, horizon + 1)
    return {"model": "chronos", "yhat": yhat, "meta": {"lookback": lookback, "horizon": horizon, "opts": opts}}
