"""Wrap script imports and execution, including launch-time failures."""
from pathlib import Path
import runpy
import sys


def launch(path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from playmodel import execution_log
    if execution_log._current is not None:
        return
    execution_log.run_logged(lambda: runpy.run_path(str(path), run_name='__main__'), program=path)
    raise SystemExit(0)
