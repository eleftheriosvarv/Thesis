
#!/usr/bin/env python3
import argparse, json
from forecast_lab.registry import list_models, run_model
from forecast_lab.dataio import load_csv, pick_default_target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True, choices=list_models() or None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--mode", choices=["zero","few","finetune"], default="zero")
    ap.add_argument("--opts", default="{}", help='JSON dict for per-model options')
    args = ap.parse_args()

    df = load_csv(args.csv)
    target = args.target or pick_default_target(df)
    cfg = {"target": target, "lookback": args.lookback, "horizon": args.horizon, "mode": args.mode, "opts": json.loads(args.opts)}
    res = run_model(args.model, df, cfg)
    print({k: (v if k != "yhat" else f"len={len(v)}") for k, v in res.items()})

if __name__ == "__main__":
    main()
