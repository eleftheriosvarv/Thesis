from typing import Dict, Any
import pandas as pd
from .preprocess.adam import AdaMParams, apply_adam

def build_cfg(df, target, lookback, horizon, mode, opts: Dict[str, Any]):
    return {
        "target": target,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "mode": mode,
        "opts": opts or {},
    }

def maybe_prefilter_adam(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    adam = (cfg.get("opts") or {}).get("adam") or {}
    if not adam.get("enabled", False):
        return df
    p = AdaMParams(
        gamma=float(adam.get("gamma", 0.10)),
        alpha=float(adam.get("alpha", 0.50)),
        beta=float(adam.get("beta", 0.50)),
        Rmin=int(adam.get("Rmin", 2)),
        Rmax=int(adam.get("Rmax", 96)),
    )
    return apply_adam(df, cfg["target"], p)
