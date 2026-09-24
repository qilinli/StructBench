"""Put data_generation/abaqus on sys.path so tests import its scripts by name."""

import sys
from pathlib import Path

ABAQUS_DIR = Path(__file__).resolve().parents[2] / "data_generation" / "abaqus"
if str(ABAQUS_DIR) not in sys.path:
    sys.path.insert(0, str(ABAQUS_DIR))
