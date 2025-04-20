from pathlib import Path

import sys
import os
import re

from datetime import datetime


def resource_path(relative_path: str) -> Path:
    """Get the absolute path to the resource, works for dev and PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Go two levels up from aegear/app.py → project root
        base_path = Path(__file__).resolve().parents[2]
    return base_path / relative_path


def get_latest_model_path(directory, model_name):
    """
    Find the latest model file in the given directory matching the base model name.
    Model files are expected to be named as: modelname_YYYY-MM-DD.pth
    """
    pattern = re.compile(rf"{re.escape(model_name)}_(\d{{4}}-\d{{2}}-\d{{2}})\.pth")
    latest_date = None
    latest_file = None

    for filename in os.listdir(directory):
        match = pattern.fullmatch(filename)
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = filename
            except ValueError:
                continue

    return os.path.join(directory, latest_file) if latest_file else None
