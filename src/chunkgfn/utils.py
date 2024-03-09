import os
from pathlib import Path


def default_data_path():
    path = Path(os.path.abspath(__file__))
    return os.path.join(str(path.parent.parent.parent), "data")
