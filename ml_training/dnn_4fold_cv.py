"""Backward-compatible command entry point for the canonical LSTM CV pipeline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lstm.lstm_model import cli
if __name__ == "__main__": cli()
