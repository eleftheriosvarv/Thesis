
import io, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import Dropdown, SelectMultiple, FileUpload, Button, HBox, VBox, Output, HTML, RadioButtons, IntText, IntSlider, Checkbox
from IPython.display import display
from .registry import list_models, get_schema, run_model

def _make_widget_from_field(name, spec):
    t = spec.get("type", "bool")
    if t == "bool":
        return Checkbox(value=bool(spec.get("default", False)), description=spec.get("label", name))
    if t == "int":
        return IntSlider(value=int(spec.get("default", 0)), min=int(spec.get("min", 0)), max=int(spec.get("max", 64)), description=spec.get("label", name), continuous_update=False)
    return IntText(value=int(spec.get("default", 0)), description=spec.get("label", name))

def launch():
    title = HTML("<h3>Forecast Lab UI</h3>")
    upload = FileUpload(accept=".csv", multiple=False)
    existing = Dropdown(options=["-- none --"] + sorted([os.path.basename(p) for p in glob.glob("*.csv")]), description="From disk:")
    load_btn = Button(description="Load CSV", button_style="info")
    out_load = Output(); file_info = HTML("<i>No CSV.</i>")

    target_dd = Dropdown(options=["--"], description="Target:")
    features_ms = SelectMultiple(options=[], description="Features:")

    model_dd = Dropdown(options=list_models(), description="Model:")
    mode_rb = RadioButtons(options=["zero", "few", "finetune"], value="zero", description="Mode:")
    lookback_in = IntText(value=96, description="Lookback")
    horizon_in  = IntText(value=24, description="Horizon")
    plotN_in    = IntText(value=400, description="Plot last N")

    # AdaM
    use_adam_cb = Checkbox(value=False, description="Use AdaM prefilter")
    gamma_in = IntText(value=10, description="gamma (x1e-2)")
    alpha_in = IntText(value=50, description="alpha (x1e-2)")
    beta_in  = IntText(value=50, description="beta (x1e-2)")
    Rmin_in  = IntText(value=2,  description="Rmin")
    Rmax_in  = IntText(value=96, description="Rmax")

    # dynamic per-model option widgets
    opts_box = VBox([]); opts_widgets = {}

    def _build_opts_ui(model_name):
        opts_widgets.clear()
        fields = get_schema(model_name)
        children = []
        for key, spec in fields.items():
            w = _make_widget_from_field(key, spec)
            opts_widgets[key] = (w, spec)
            children.append(w)
        opts_box.children = children

    _build_opts_ui(model_dd.value)

    out_run = Output(); run_btn = Button(description="Run", button_style="primary")
    STATE = {"df": None}

    def on_load(_):
        out_load.clear_output()
        with out_load:
            df = None
            if upload.value:
                key = list(upload.value.keys())[0]
                content = upload.value[key]["content"]
                name = upload.value[key]["metadata"]["name"]
                df = pd.read_csv(io.BytesIO(content)); df.to_csv(name, index=False)
                file_info.value = f"<b>Loaded (upload):</b> {name} shape={df.shape}"
            elif existing.value and existing.value != "-- none --":
                df = pd.read_csv(existing.value)
                file_info.value = f"<b>Loaded (disk):</b> {existing.value} shape={df.shape}"
            else:
                file_info.value = "<i>No CSV selected.</i>"; return
            STATE["df"] = df
            cols = list(df.columns)
            target_dd.options = cols
            nums = df.select_dtypes("number").columns
            target_dd.value = nums[0] if len(nums) else cols[0]
            features_ms.options = [c for c in cols if c != target_dd.value]
            display(df.head())

    def on_model_changed(change):
        _build_opts_ui(change["new"])

    def on_run(_):
        out_run.clear_output()
        with out_run:
            if STATE["df"] is None:
                print("Load a CSV first"); return
            target = target_dd.value
            lookback = int(lookback_in.value); horizon = int(horizon_in.value)
            mode = mode_rb.value
            # collect per-model opts with simple depends_on
            tmp_vals = {k: w[0].value for k, w in opts_widgets.items()}
            opts = {}
            for k, (w, spec) in opts_widgets.items():
                dep = spec.get("depends_on")
                if dep and not tmp_vals.get(dep, False):
                    continue
                opts[k] = w.value
            # AdaM opts
            opts["adam"] = {
                "enabled": bool(use_adam_cb.value),
                "gamma": gamma_in.value / 100.0,
                "alpha": alpha_in.value / 100.0,
                "beta":  beta_in.value  / 100.0,
                "Rmin": Rmin_in.value,
                "Rmax": Rmax_in.value,
            }
            cfg = {"target": target, "lookback": lookback, "horizon": horizon, "mode": mode, "opts": opts}
            res = run_model(model_dd.value, STATE["df"], cfg)
            y = pd.to_numeric(STATE["df"][target], errors="coerce").dropna().to_numpy(float)
            H = len(res["yhat"]); yplot = y[-max(int(plotN_in.value), 2):]
            print("y:", len(y), "| yhat:", H, "| opts:", {k:v for k,v in opts.items() if k!='adam' or v.get('enabled')})
            plt.figure(figsize=(10,4))
            plt.plot(np.arange(len(yplot)), yplot, label="y (last N)")
            plt.plot(np.arange(len(yplot)-1, len(yplot)-1+H), res["yhat"], label=f"yhat ({H})")
            plt.legend(); plt.title(f"{model_dd.value} — {mode}")
            plt.show()

    load_btn.on_click(on_load)
    model_dd.observe(on_model_changed, names="value")
    run_btn.on_click(on_run)

    layout = VBox([
        title,
        VBox([HTML("<b>1) Pick CSV</b>"), HBox([upload, existing, load_btn]), file_info, out_load]),
        VBox([HTML("<b>2) Pick columns</b>"), HBox([target_dd, features_ms])]),
        VBox([HTML("<b>3) Pick model & mode</b>"), HBox([model_dd, mode_rb])]),
        VBox([HTML("<b>4) AdaM (optional)</b>"), HBox([use_adam_cb, gamma_in, alpha_in, beta_in, Rmin_in, Rmax_in])]),
        VBox([HTML("<b>5) Model options</b>"), opts_box]),
        VBox([HTML("<b>6) Params</b>"), HBox([lookback_in, horizon_in, plotN_in])]),
        VBox([HTML("<b>7) Run</b>"), run_btn, out_run]),
    ])
    display(layout)
