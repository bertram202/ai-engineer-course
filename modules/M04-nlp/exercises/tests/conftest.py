"""Тесты проверяют ваш код из exercises/. CI проверяет эталон: MLCOURSE_TARGET=solutions."""

import os
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[2]
TARGET = os.environ.get("MLCOURSE_TARGET", "exercises")
sys.path.insert(0, str(MODULE_DIR / TARGET))
