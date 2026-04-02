import importlib
import pkgutil
import os

_pkg_dir = os.path.dirname(__file__)
for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    importlib.import_module(f"{__name__}.{_name}")
