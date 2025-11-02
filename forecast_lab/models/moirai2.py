# forecast_lab/models/moirai2.py
import os
import time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Optional metrics imports (comment out if not available) ----
try:
    from ..metrics import mae, rmse, smape_pct, r2, summary_df, save_summary_csv
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False
    def mae(y, yhat): return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
    def rmse(y, yhat): 
        e = np.asarray(y) - np.asarray(yhat)
        return float(np.sqrt(np.mean(e**2)))
    def smape_pct(y, yhat):
        y = np.asarray(y, float); yhat = np.asarray(yhat, float)
        denom = (np.abs(y) + np.abs(yhat)).clip(1e-8, None)
        return float(np.mean(2.0 * np.abs(yhat - y) / denom) * 100.0)
    def r2(y, yhat):
        y = np.asarray(y, float); yhat = np.asarray(yhat, float)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        return float(1.0 - ss_res / ss_tot)

def _lazy_import_moirai2():
    from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast
    return Moirai2Module, Moirai2Forecast

SCHEMA = {
    "use_adam": {"type": "bool", "default": False, "label": "Pre-filter with AdaM"},
    "timestamp_col": {"type": "str", "default": None, "label": "Timestamp column (auto if None)"},
    "lookback": {"type": "int", "default": 512},
    "horizon": {"type": "int", "default": 96},
    "plot": {"type": "bool", "default": True},
    "save": {"type": "bool", "default": True},
    "hf_token": {"type": "str", "default": None},
    "hf_model": {"type": "str", "default": "Salesforce/moirai-2.0-R-small"}
}

def get_schema() -> Dict[str, Any]:
    return SCHEMA

def _auto_timestamp_col(df: pd.DataFrame, user_hint: Optional[str]) -> Optional[str]:
    if user_hint and user_hint in df.columns:
        return user_hint
    for c in ["timestamp", "date", "datetime", "time", "ts"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return None

def _ensure_results_dir() -> str:
    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def run(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Zero-shot: take last LOOKBACK points of target and forecast HORIZON ahead with Moirai-2.
    If AdaM is enabled, the controller should have pre-filtered df before calling this.
    Saves metrics/plot to results/ when enabled.
    """
    lookback = int(cfg.get("lookback", 512))
    horizon  = int(cfg.get("horizon", 96))
    target   = cfg["target"]
    ts_col   = _auto_timestamp_col(df, cfg.get("timestamp_col"))
    save_out = bool(cfg.get("save", True))
    do_plot  = bool(cfg.get("plot", True))
    hf_token = cfg.get("hf_token", None)
    hf_model = cfg.get("hf_model", "Salesforce/moirai-2.0-R-small")

    if ts_col:
        df = df.sort_values(ts_col).reset_index(drop=True)

    y_all = pd.to_numeric(df[target], errors="coerce").astype("float32").to_numpy()
    y_all = y_all[~np.isnan(y_all)]
    if len(y_all) < lookback:
        raise ValueError(f"Series too short: {len(y_all)} < lookback={lookback}")

    ctx = y_all[-lookback:].astype(np.float32, copy=False)

    # Build predictor (lazy heavy import)
    Moirai2Module, Moirai2Forecast = _lazy_import_moirai2()
    module = Moirai2Module.from_pretrained(pretrained_model_name_or_path=hf_model, token=hf_token)
    predictor = Moirai2Forecast(
        module=module,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=lookback,
        prediction_length=horizon,
    )

    t0 = time.time()
    try:
        out = predictor.predict_quantile(past_target=[ctx], quantile_levels=[0.5])
        first = next(iter(out))
        try:
            yhat = np.asarray(first.quantile(0.5), np.float32).reshape(-1)
        except Exception:
            m = first.mean if not callable(first.mean) else first.mean()
            yhat = np.asarray(m, np.float32).reshape(-1)
    except Exception:
        out = predictor.predict(past_target=[ctx])
        first = next(iter(out))
        try:
            m = first.mean if not callable(first.mean) else first.mean()
            yhat = np.asarray(m, np.float32).reshape(-1)
        except Exception:
            yhat = np.asarray(first.samples, np.float32).mean(0).reshape(-1)
    runtime_s = float(time.time() - t0)

    yhat = yhat[:horizon]

    # quick tail evaluation: compare yhat vs last part of context (80/20 split of lookback)
    split = max(1, int(0.8 * lookback))
    eval_ctx = ctx[split:]
    n_eval = int(min(len(eval_ctx), len(yhat)))
    true_eval = eval_ctx[:n_eval]
    pred_eval = yhat[:n_eval]

    M_MAE = mae(true_eval, pred_eval) if n_eval > 0 else float("nan")
    M_RMSE = rmse(true_eval, pred_eval) if n_eval > 0 else float("nan")
    M_SMAPE = smape_pct(true_eval, pred_eval) if n_eval > 0 else float("nan")
    M_R2 = r2(true_eval, pred_eval) if n_eval > 0 else float("nan")

    out_dir = _ensure_results_dir()
    tag = f"moirai2_{target}_L{lookback}_H{horizon}"

    # Save CSVs
    if save_out:
        if _HAS_METRICS:
            df_sum = summary_df("FULL|zero-shot", n_eval, 100.0, M_MAE, M_RMSE, M_SMAPE, M_R2,
                                acc_pct=None, acc_delta=None, mase_m=lookback, runtime_s=runtime_s,
                                model="moirai2")
            save_summary_csv(df_sum, "moirai2", out_dir)
        else:
            pd.DataFrame([{
                "model": "moirai2",
                "n_eval": n_eval,
                "lookback": lookback,
                "horizon": horizon,
                "MAE": M_MAE,
                "RMSE": M_RMSE,
                "sMAPE_%": M_SMAPE,
                "R2": M_R2,
                "runtime_s": runtime_s
            }]).to_csv(os.path.join(out_dir, "moirai2_metrics_summary.csv"), index=False)
        pd.DataFrame({"yhat": yhat}).to_csv(os.path.join(out_dir, f"{tag}_yhat.csv"), index=False)

    # Save plot
    if do_plot:
        plt.figure(figsize=(11, 3.5))
        x_ctx = np.arange(len(ctx))
        x_fut = np.arange(len(ctx), len(ctx) + len(yhat))
        plt.plot(x_ctx, ctx, label="Context")
        plt.plot(x_fut, yhat, label="Forecast")
        plt.title(f"Moirai-2 | target={target} | L={lookback} H={horizon}")
        plt.legend()
        plt.tight_layout()
        if save_out:
            plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=120)
        plt.close()

    return {
        "model": "moirai2",
        "yhat": yhat,
        "meta": {
            "lookback": lookback,
            "horizon": horizon,
            "runtime_s": runtime_s,
            "n_eval": n_eval,
            "metrics": {"MAE": M_MAE, "RMSE": M_RMSE, "sMAPE_%": M_SMAPE, "R2": M_R2},
            "used_adam": bool(cfg.get("use_adam", False)),
            "timestamp_col": ts_col,
            "hf_model": hf_model
        }
    }

