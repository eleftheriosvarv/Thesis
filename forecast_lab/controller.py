
def build_cfg(df, target, lookback, horizon, mode, opts: dict):
    return {
        "target": target,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "mode": mode,
        "opts": opts or {},
    }
