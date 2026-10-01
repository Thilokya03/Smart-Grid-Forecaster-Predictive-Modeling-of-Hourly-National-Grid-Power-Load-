import sys
import os
from pathlib import Path

# Add the project root to sys.path so that tests can import modules like ui and weather_pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
