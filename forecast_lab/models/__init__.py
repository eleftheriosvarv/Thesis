import importlib
import pkgutil
import pathlib

_pkg_path = pathlib.Path(__file__).parent

for _, name, _ in pkgutil.iter_modules([str(_pkg_path)]):
    if name.startswith("_"):
        continue
    importlib.import_module(f"forecast_lab.models.{name}")

