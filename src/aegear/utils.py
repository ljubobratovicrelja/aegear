from pathlib import Path
import sys


def resource_path(relative_path: str) -> Path:
    """Get the absolute path to the resource, works for dev and PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Go two levels up from aegear/app.py → project root
        base_path = Path(__file__).resolve().parents[2]
    return base_path / relative_path
