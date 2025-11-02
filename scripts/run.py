#!/usr/bin/env python3
import argparse
import json
import re
import ast
from forecast_lab.registry import list_models, run_model
from forecast_lab.dataio import load_csv, pick_default_target
from forecast_lab.controller import build_cfg, maybe_prefilter_adam


def parse_opts(s: str) -> dict:
    """
    Robust options parser:
    - Accepts normal JSON:           {"timestamp_col": 0}
    - Accepts Python-style dicts:    {'timestamp_col': 0}
    - Accepts unquoted keys:         {timestamp_col: 0}
    Returns {} on empty/None.
    Raises last JSON error if nothing works.
    """
    if not s:
        return {}
    # 1) Try JSON as-is
    try:
        return json.loads(s)
    except Exception:
        pass
    # 2) Try Python literal (single quotes, etc.)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # 3) Auto-quote unquoted keys, normalize quotes, then JSON-decode
    s2 = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', s)
    s2 = s2.replace("'", '"')
    return json.loads(s2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', required=True, choices=list_models() or None)
    ap.add_argument('--target', default=None)
    ap.add_argument('--lookback', type=int, default=96)
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--mode', choices=['zero', 'few', 'finetune'], default='zero')
    ap.add_argument('--opts', default='{}',
                    help='Model/runtime options. Accepts JSON or Python-style dict '
                         '(e.g., {"adam": {"enabled": true}} or {adam:{enabled:true}}).')
    ap.add_argument('--opts_path', default=None,
                    help='Path to a JSON file with options (overrides --opts if provided).')
    args = ap.parse_args()

    # Load data and target
    df = load_csv(args.csv)
    target = args.target or pick_default_target(df)

    # Parse options (file > inline)
    if args.opts_path:
        with open(args.opts_path, 'r', encoding='utf-8') as f:
            opts = json.load(f)
    else:
        opts = parse_opts(args.opts)

    # Build config and (optionally) prefilter via AdaM
    cfg = build_cfg(df, target, args.lookback, args.horizon, args.mode, opts)
    df_run = maybe_prefilter_adam(df, cfg)

    # Run selected model
    res = run_model(args.model, df_run, cfg)

    # Compact print (avoid dumping full yhat)
    print({k: (v if k != 'yhat' else f'len={len(v)}') for k, v in res.items()})


if __name__ == '__main__':
    main()

