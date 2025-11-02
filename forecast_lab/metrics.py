import os, numpy as np, pandas as pd

def mae(y, yhat):  y, yhat = np.asarray(y,float), np.asarray(yhat,float); return float(np.mean(np.abs(y-yhat)))
def rmse(y, yhat): y, yhat = np.asarray(y,float), np.asarray(yhat,float); return float(np.sqrt(np.mean((y-yhat)**2)))
def smape(y, yhat):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float)
    return float(100.0*np.mean(2*np.abs(y-yhat)/((np.abs(y)+np.abs(yhat))+1e-12)))
def r2(y, yhat):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float)
    ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-np.mean(y))**2)+1e-12
    return float(1.0-ss_res/ss_tot)

def acc_at_pct(y, yhat, p=5.0):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float); tol = (p/100.0)*(np.abs(y)+1e-12)
    return float(100.0*np.mean(np.abs(y-yhat) <= tol))

def acc_at_delta(y, yhat, delta=100.0):
    y, yhat = np.asarray(y,float), np.asarray(yhat,float)
    return float(100.0*np.mean(np.abs(y-yhat) <= float(delta)))

def evaluate_all(y_true, y_pred, pct=5.0, delta=100.0):
    return {
        "MAE": mae(y_true,y_pred),
        "RMSE": rmse(y_true,y_pred),
        "sMAPE_%": smape(y_true,y_pred),
        "R2": r2(y_true,y_pred),
        "Acc@±5%_%": acc_at_pct(y_true,y_pred,pct),
        "Acc@±100_%": acc_at_delta(y_true,y_pred,delta),
    }

def summary_df(model_name, y_true, y_pred, runtime_s, tag="FULL", coverage=100.0):
    m = evaluate_all(y_true,y_pred)
    return pd.DataFrame([{
        "tag": tag, "n_eval": int(len(y_true)), "coverage_%": float(coverage),
        **m, "runtime_s": float(runtime_s), "model": model_name
    }])

def save_summary_csv(df_summary: pd.DataFrame, model_name: str, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_metrics_summary.csv")
    mode, header = ("a", False) if os.path.exists(path) else ("w", True)
    df_summary.to_csv(path, index=False, mode=mode, header=header)
    return path
