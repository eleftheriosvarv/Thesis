from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class AdaMParams:
    gamma: float = 0.10
    alpha: float = 0.50
    beta:  float = 0.50
    Rmin:  int   = 2
    Rmax:  int   = 96

def apply_adam(df: pd.DataFrame, target: str, p: AdaMParams) -> pd.DataFrame:
    s = pd.to_numeric(df[target], errors="coerce").astype(float)
    s = s.fillna(method="ffill").fillna(method="bfill")
    if s.size < 3:
        return df
    delta = s.diff().abs()
    thr = p.gamma * (s.abs().median() + 1e-9)
    keep = delta.fillna(0) > thr
    keep.iloc[0] = True
    kept_df = df.loc[keep].reset_index(drop=True)
    return kept_df if len(kept_df) >= p.Rmin else df
