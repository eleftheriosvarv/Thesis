# forecast_lab/models/moirai2.py
import os, time
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _auto_timestamp_col(df: pd.DataFrame, user_hint: Optional[str]) -> Optional[str]:
    if user_hint and user_hint in df.columns: return user_hint
    for c in ("timestamp","date","datetime","time","ts"):
        if c in df.columns: return c
    for c in df.columns:
        try:
            if np.issubdtype(df[c].dtype, np.datetime64): return c
        except Exception:
            pass
    return None

def get_schema() -> Dict[str, Any]:
    return {
        "use_adam":      {"type": "bool", "default": False, "label": "Pre-filter with AdaM"},
        "timestamp_col": {"type": "str",  "default": None,  "label": "Timestamp column (auto if None)"},
        "lookback":      {"type": "int",  "default": 512},
        "horizon":       {"type": "int",  "default": 96},
        "plot":          {"type": "bool", "default": True},
        "save":          {"type": "bool", "default": True},
        "hf_token":      {"type": "str",  "default": None},
        "hf_model":      {"type": "str",  "default": "Salesforce/moirai-2.0-R-small"},
    }

def _ensure_results_dir() -> str:
    out = "results"
    os.makedirs(out, exist_ok=True)
    return out

# minimal numpy-only metrics (no extra deps)
def _mae(y, yhat):  y=np.asarray(y); yhat=np.asarray(yhat); return float(np.mean(np.abs(y-yhat)))
def _rmse(y, yhat): y=np.asarray(y); yhat=np.asarray(yhat); return float(np.sqrt(np.mean((y-yhat)**2)))
def _smape(y, yhat):
    y=np.asarray(y); yhat=np.asarray(yhat)
    denom = (np.abs(y)+np.abs(yhat))/2.0
    return float(np.mean(np.where(denom>1e-8, np.abs(y-yhat)/denom, 0.0))*100.0)
def _r2(y, yhat):
    y=np.asarray(y); yhat=np.asarray(yhat)
    ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-y.mean())**2)+1e-12
    return float(1.0 - ss_res/ss_tot)

def _lazy_import_moirai2():
    from uni2ts.model.moirai2 import Moirai2Module, Moirai2Forecast
    return Moirai2Module, Moirai2Forecast

def run(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
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
        f0  = next(iter(out))
        try:
            yhat = np.asarray(f0.quantile(0.5), dtype=np.float32).reshape(-1)
        except Exception:
            meanv = f0.mean if not callable(f0.mean) else f0.mean()
            yhat = np.asarray(meanv, dtype=np.float32).reshape(-1)
    except Exception:
        out = predictor.predict(past_target=[ctx])
        f0  = next(iter(out))
        try:
            meanv = f0.mean if not callable(f0.mean) else f0.mean()
            yhat = np.asarray(meanv, dtype=np.float32).reshape(-1)
        except Exception:
            yhat = np.asarray(f0.samples, dtype=np.float32).mean(axis=0).reshape(-1)
    runtime_s = float(time.time() - t0)

    yhat = yhat[:horizon]

    # quick 80/20 split on context for comparable tail metric
    split = max(1, int(0.8*lookback))
    eval_ctx = ctx[split:]
    n_eval = int(min(len(eval_ctx), len(yhat)))
    true_eval = eval_ctx[:n_eval]
    pred_eval = yhat[:n_eval]

    M_MAE   = _mae(true_eval, pred_eval)   if n_eval>0 else float("nan")
    M_RMSE  = _rmse(true_eval, pred_eval)  if n_eval>0 else float("nan")
    M_SMAPE = _smape(true_eval, pred_eval) if n_eval>0 else float("nan")
    M_R2    = _r2(true_eval, pred_eval)    if n_eval>1 else float("nan")

    out_dir = _ensure_results_dir()
    tag = f"moirai2_{target}_L{lookback}_H{horizon}"

    if do_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np
            x_ctx = _np.arange(len(ctx))
            x_fut = _np.arange(len(ctx), len(ctx)+len(yhat))
            plt.figure(figsize=(11,3.5))
            plt.plot(x_ctx, ctx, label="Context")
            plt.plot(x_fut, yhat, label="Forecast")
            plt.title(f"Moirai-2 | target={target} | L={lookback} H={horizon}")
            plt.legend(); plt.tight_layout()
            if save_out:
                plt.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=120)
            plt.close()
        except Exception:
            pass

    if save_out:
        pd.DataFrame({"yhat": yhat}).to_csv(os.path.join(out_dir, f"{tag}_yhat.csv"), index=False)
        pd.DataFrame([{
            "model":"moirai2","n_eval":n_eval,"lookback":lookback,"horizon":horizon,
            "MAE":M_MAE,"RMSE":M_RMSE,"sMAPE_%":M_SMAPE,"R2":M_R2,"runtime_s":runtime_s
        }]).to_csv(os.path.join(out_dir, "moirai2_metrics_summary.csv"), index=False)

    return {
        "model":"moirai2",
        "yhat": yhat,
        "meta":{
            "lookback":lookback,"horizon":horizon,"runtime_s":runtime_s,"n_eval":n_eval,
            "metrics":{"MAE":M_MAE,"RMSE":M_RMSE,"sMAPE_%":M_SMAPE,"R2":M_R2},
            "used_adam": bool(cfg.get("use_adam", False)),
            "timestamp_col": ts_col,
            "hf_model": hf_model
        }
    }