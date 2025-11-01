
from forecast_lab.registry import register_model, list_models, run_model

def test_register_and_run():
    @register_model("dummy", schema={"flag": {"type": "bool", "default": False}})
    def run_dummy(df, cfg):
        return {"model": "dummy", "yhat": [1,2,3], "meta": cfg.get("opts", {})}

    assert "dummy" in list_models()
    out = run_model("dummy", {"value": [0]}, {"target": "value", "opts": {"flag": True}})
    assert out["model"] == "dummy"
    assert len(out["yhat"]) == 3
