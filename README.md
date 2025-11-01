
# forecast-lab

Modular forecasting experiments with a central controller, model registry, and notebook UI.

## Install
```bash
pip install -e .
# or with extras:
pip install -e .[chronos,timesfm]
```

## Notebook
```python
from forecast_lab.ui_widgets import launch
launch()
```

## CLI
```bash
fl-run --csv your.csv --model chronos --target value --lookback 96 --horizon 24 --mode zero --opts '{}'
```
