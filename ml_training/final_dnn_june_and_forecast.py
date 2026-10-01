"""Backward-compatible command entry point for final locked-June LSTM testing."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.lstm.final_dnn_june_and_forecast import main
if __name__ == "__main__": main()
