
#!/usr/bin/env python3
from forecast_lab.registry import list_models, run_model
from forecast_lab.dataio import load_csv, pick_default_target

CSV = "data.csv"  # change per use
LOOKBACK, HORIZON = 96, 24

df = load_csv(CSV)
target = pick_default_target(df)

for name in list_models():
    res = run_model(name, df, {"target": target, "lookback": LOOKBACK, "horizon": HORIZON, "mode": "zero", "opts": {}})
    print(f"{name:16s} -> yhat={len(res['yhat'])}")
