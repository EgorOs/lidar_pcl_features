import os
from pathlib import Path

PROJ_ROOT = Path(os.getenv('PROJ_ROOT', Path(__file__).resolve().parents[1]))
