"""LSTM and Transformer encoders with a direct 168-output head.

All four models here emit the whole 168-hour forecast in one forward pass
(direct multi-output), so there is no recursion and no error compounding, and
no model ever needs a future demand lag it does not have.

Leakage controls that apply to every model in this file:

* The StandardScaler for demand and the StandardScaler for weather are fitted
  on the fold's training rows only, inside `fit`, and reused unchanged at
  prediction time.
* Early stopping selects the epoch on the inner-validation window tail of the
  training range, which ends before the evaluation period begins. A gap of 168
  window origins separates the training and inner-validation sets, so no
  inner-validation target hour was ever a training target hour.
* The `_features` variants see observed weather in the encoder only. The
  decoder side receives calendar attributes of the 168 target hours and
  nothing else -- no future weather, no future demand.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pandas as pd

from . import protocol as P
from .base import ForecastModel

HIDDEN_SIZE = 64
DENSE_SIZE = 256
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 20

D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2
FF_DIM = 128
TRANSFORMER_DROPOUT = 0.1


#: Global torch settings that change numerics. They are pinned here, in one
#: place every torch-based model in this harness routes through, because
#: TimesFM's loader also calls `set_float32_matmul_precision("high")` -- so
#: without pinning, a model's output depended on whether TimesFM had been
#: loaded earlier in the same process. The behavioural leakage probe caught
#: exactly that as a 95 MW phantom "leak".
MATMUL_PRECISION = "high"


def set_seed(seed: int = P.SEED) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    if torch.cuda.is_available():
        # The fused attention kernels have non-deterministic backward passes,
        # which left the Transformer models irreproducible run to run (up to
        # ~440 MW on identical input). The math kernel is deterministic.
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
def _build_modules():
    import torch
    from torch import nn

    class DirectLSTM(nn.Module):
        """LSTM encoder, optional flattened known-future calendar, 168 outputs."""

        def __init__(self, encoder_channels: int, future_channels: int = 0):
            super().__init__()
            self.lstm = nn.LSTM(encoder_channels, HIDDEN_SIZE, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(DROPOUT)
            head_in = HIDDEN_SIZE + P.FORECAST_HORIZON * future_channels
            self.fc1 = nn.Linear(head_in, DENSE_SIZE)
            self.fc2 = nn.Linear(DENSE_SIZE, P.FORECAST_HORIZON)
            self.relu = nn.ReLU()

        def forward(self, encoder, future=None):
            output, _ = self.lstm(encoder)
            hidden = output[:, -1, :]
            if future is not None:
                hidden = torch.cat([hidden, future.flatten(1)], dim=1)
            return self.fc2(self.relu(self.fc1(self.dropout(hidden))))

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 1024):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class DirectTransformer(nn.Module):
        """Transformer encoder (C11 style), optional calendar, 168 outputs."""

        def __init__(self, encoder_channels: int, future_channels: int = 0):
            super().__init__()
            self.project = nn.Linear(encoder_channels, D_MODEL)
            self.positional = PositionalEncoding(D_MODEL)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FF_DIM,
                dropout=TRANSFORMER_DROPOUT, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
            self.dropout = nn.Dropout(DROPOUT)
            head_in = D_MODEL + P.FORECAST_HORIZON * future_channels
            self.fc1 = nn.Linear(head_in, DENSE_SIZE)
            self.fc2 = nn.Linear(DENSE_SIZE, P.FORECAST_HORIZON)
            self.relu = nn.ReLU()

        def forward(self, encoder, future=None):
            x = self.positional(self.project(encoder))
            x = self.encoder(x)
            hidden = x[:, -1, :]
            if future is not None:
                hidden = torch.cat([hidden, future.flatten(1)], dim=1)
            return self.fc2(self.relu(self.fc1(self.dropout(hidden))))

    return DirectLSTM, DirectTransformer


# ---------------------------------------------------------------------------
# Shared training / prediction
# ---------------------------------------------------------------------------
class _NeuralForecaster(ForecastModel):
    use_features = False
    architecture = "lstm"      # or "transformer"

    def __init__(self, calendar_frame: pd.DataFrame | None = None):
        self._calendar_frame = calendar_frame
        self._model = None
        self._demand_scaler = None
        self._weather_scaler = None
        self.best_epoch = -1
        self.n_train_windows = 0
        self.n_inner_windows = 0

    # -- tensors ---------------------------------------------------------
    def _arrays(self, frame: pd.DataFrame, fit_scalers: bool, train_rows: int | None = None):
        from sklearn.preprocessing import StandardScaler

        demand = frame["demand_mw"].to_numpy(np.float64).reshape(-1, 1)
        if fit_scalers:
            limit = train_rows if train_rows is not None else len(frame)
            self._demand_scaler = StandardScaler().fit(demand[:limit])
        demand_scaled = self._demand_scaler.transform(demand).astype(np.float32)

        channels = [demand_scaled]
        if self.use_features:
            weather = P.past_weather_features(frame).to_numpy(np.float64)
            if fit_scalers:
                limit = train_rows if train_rows is not None else len(frame)
                self._weather_scaler = StandardScaler().fit(weather[:limit])
            channels.append(self._weather_scaler.transform(weather).astype(np.float32))
            stamps = pd.DatetimeIndex(frame["timestamp"])
            source = self._calendar_frame if self._calendar_frame is not None else frame
            channels.append(P.calendar_features(stamps, source).to_numpy(np.float32))
        encoder = np.concatenate(channels, axis=1)

        future_calendar = None
        if self.use_features:
            stamps = pd.DatetimeIndex(frame["timestamp"])
            extended = stamps.append(
                pd.date_range(stamps[-1] + pd.Timedelta(hours=1),
                              periods=P.FORECAST_HORIZON, freq="h")
            )
            source = self._calendar_frame if self._calendar_frame is not None else frame
            future_calendar = P.calendar_features(extended, source).to_numpy(np.float32)
        return encoder, demand_scaled[:, 0], future_calendar

    def _windows(self, encoder, target, future_calendar, origins, dev):
        import torch

        back = np.arange(-P.INPUT_LENGTH + 1, 1)
        ahead = np.arange(1, P.FORECAST_HORIZON + 1)
        enc_idx = origins[:, None] + back[None, :]
        tgt_idx = origins[:, None] + ahead[None, :]
        x = torch.from_numpy(encoder[enc_idx]).to(dev)
        y = torch.from_numpy(target[tgt_idx]).to(dev)
        f = None
        if future_calendar is not None:
            f = torch.from_numpy(future_calendar[tgt_idx]).to(dev)
        return x, f, y

    # -- fit -------------------------------------------------------------
    def fit(self, train_frame: pd.DataFrame) -> None:
        import torch
        from torch import nn

        set_seed()
        dev = device()
        DirectLSTM, DirectTransformer = _build_modules()

        origins = P.window_origin_indices(len(train_frame))
        train_origins, inner_origins = P.inner_split(origins)
        if len(train_origins) == 0 or len(inner_origins) == 0:
            raise RuntimeError(f"{self.name}: empty train or inner-validation window set")
        self.n_train_windows = int(len(train_origins))
        self.n_inner_windows = int(len(inner_origins))

        # Scalers see only rows that are training-window inputs or targets --
        # i.e. rows strictly before the inner-validation region begins.
        scaler_rows = int(train_origins[-1] + P.FORECAST_HORIZON + 1)
        encoder, target, future_calendar = self._arrays(
            train_frame, fit_scalers=True, train_rows=scaler_rows
        )

        xt, ft, yt = self._windows(encoder, target, future_calendar, train_origins, dev)
        xi, fi, yi = self._windows(encoder, target, future_calendar, inner_origins, dev)

        future_channels = 0 if future_calendar is None else future_calendar.shape[1]
        cls = DirectLSTM if self.architecture == "lstm" else DirectTransformer
        model = cls(encoder.shape[1], future_channels).to(dev)
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        best_loss, best_state, waited = float("inf"), None, 0
        generator = torch.Generator(device="cpu").manual_seed(P.SEED)
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = torch.randperm(len(xt), generator=generator).to(dev)
            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start:start + BATCH_SIZE]
                optimiser.zero_grad()
                output = model(xt[batch], None if ft is None else ft[batch])
                loss = criterion(output, yt[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            model.eval()
            with torch.no_grad():
                inner_loss = float(criterion(model(xi, fi), yi))
            if inner_loss < best_loss - 1e-9:
                best_loss, waited = inner_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                self.best_epoch = epoch
            else:
                waited += 1
                if waited >= PATIENCE:
                    break
        if best_state is None:
            raise RuntimeError(f"{self.name}: training produced no usable epoch")
        model.load_state_dict(best_state)
        model.eval()
        self._model = model

    # -- predict ---------------------------------------------------------
    def predict(self, origin, history, future_calendar_frame) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError(f"{self.name} was not fitted")
        encoder, _, future_calendar = self._arrays(history, fit_scalers=False)
        dev = device()
        x = torch.from_numpy(encoder[-P.INPUT_LENGTH:][None, ...]).to(dev)
        f = None
        if future_calendar is not None:
            start = len(history)
            f = torch.from_numpy(
                future_calendar[start:start + P.FORECAST_HORIZON][None, ...]
            ).to(dev)
        with torch.no_grad():
            scaled = self._model(x, f).cpu().numpy().reshape(-1, 1)
        return self._demand_scaler.inverse_transform(scaled).ravel().astype(float)


_COMMON_TUNING = (
    f"fixed: hidden {HIDDEN_SIZE}, dense {DENSE_SIZE}, dropout {DROPOUT}, "
    f"lr {LEARNING_RATE}, batch {BATCH_SIZE}; epoch early-stopped "
    f"(patience {PATIENCE}, cap {MAX_EPOCHS}) on the inner-validation tail only"
)
_TRANSFORMER_TUNING = (
    f"fixed: d_model {D_MODEL}, {N_HEAD} heads, {N_LAYERS} layers, ff {FF_DIM}, "
    f"lr {LEARNING_RATE}, batch {BATCH_SIZE}; epoch early-stopped "
    f"(patience {PATIENCE}, cap {MAX_EPOCHS}) on the inner-validation tail only"
)


class LSTMDirect(_NeuralForecaster):
    name = "lstm"
    architecture = "lstm"
    use_features = False
    inputs = "past demand only (168h)"
    method = "direct multi-output (168 outputs in one pass)"
    tuning = _COMMON_TUNING


class LSTMFeatures(_NeuralForecaster):
    name = "lstm_features"
    architecture = "lstm"
    use_features = True
    inputs = "past demand + observed past weather + calendar (168h encoder), known-future calendar (168h decoder)"
    method = "direct multi-output (168 outputs in one pass)"
    tuning = _COMMON_TUNING


class TransformerDirect(_NeuralForecaster):
    name = "transformer"
    architecture = "transformer"
    use_features = False
    inputs = "past demand only (168h)"
    method = "direct multi-output (168 outputs in one pass)"
    tuning = _TRANSFORMER_TUNING


class TransformerFeatures(_NeuralForecaster):
    name = "transformer_features"
    architecture = "transformer"
    use_features = True
    inputs = "past demand + observed past weather + calendar (168h encoder), known-future calendar (168h decoder)"
    method = "direct multi-output (168 outputs in one pass)"
    tuning = _TRANSFORMER_TUNING
