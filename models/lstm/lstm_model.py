"""Leakage-safe four-fold LSTM validation for hourly electricity demand."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from models.cross_validation import FINAL_TEST_START, VALIDATION_FOLDS, validate_folds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_training_data.csv"
# Canonical lower-case location; safe on Linux and Windows.
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dnn" / "dnn_outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "dnn" / "checkpoints"
PREDICTIONS_PATH, ALL_HORIZONS_PATH = OUTPUT_DIR / "dnn_predictions.csv", OUTPUT_DIR / "dnn_predictions_all_horizons.csv"
HORIZON_METRICS_PATH, METRICS_PATH = OUTPUT_DIR / "dnn_metrics_by_horizon.csv", OUTPUT_DIR / "dnn_metrics.json"
FOLD_METRICS_PATH = OUTPUT_DIR / "dnn_validation_metrics.csv"
SEED, INPUT_LENGTH, FORECAST_HORIZON = 42, 168, 24
HIDDEN_SIZE, DENSE_SIZE, DROPOUT = 64, 32, .2
BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE, INNER_VALIDATION_HOURS = 64, 15, 5, .005, 168
FOLDS = VALIDATION_FOLDS

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError: pass

class LoadForecastDataset(Dataset):
    def __init__(self, x_values, y_values): self.x_values=torch.as_tensor(x_values,dtype=torch.float32); self.y_values=torch.as_tensor(y_values,dtype=torch.float32)
    def __len__(self): return len(self.x_values)
    def __getitem__(self, index): return self.x_values[index],self.y_values[index]

class BaselineLSTM(nn.Module):
    def __init__(self,input_size=1,hidden_size=HIDDEN_SIZE,dense_size=DENSE_SIZE,forecast_horizon=FORECAST_HORIZON,dropout=DROPOUT):
        super().__init__(); self.lstm=nn.LSTM(input_size,hidden_size,num_layers=1,batch_first=True); self.dropout=nn.Dropout(dropout); self.fc1=nn.Linear(hidden_size,dense_size); self.relu=nn.ReLU(); self.fc2=nn.Linear(dense_size,forecast_horizon)
    def forward(self,x):
        output,_=self.lstm(x); return self.fc2(self.relu(self.fc1(self.dropout(output[:,-1,:]))))

def load_data(path=MASTER_PATH):
    data=pd.read_csv(path,low_memory=False)
    if not {"timestamp","demand_mw"}.issubset(data): raise ValueError("Master data requires timestamp and demand_mw.")
    data["timestamp"]=pd.to_datetime(data.timestamp,errors="coerce"); data["demand_mw"]=pd.to_numeric(data.demand_mw,errors="coerce")
    if data[["timestamp","demand_mw"]].isna().any().any(): raise ValueError("timestamps must parse and demand_mw must be numeric; clean invalid rows upstream.")
    if data.timestamp.duplicated().any(): raise ValueError("Duplicate timestamps found; resolve them upstream.")
    return data.sort_values("timestamp").reset_index(drop=True)

def create_continuous_sequences(values,timestamps):
    """Build t-167..t inputs and t+1..t+24 targets, skipping any missing hour."""
    time=pd.DatetimeIndex(pd.to_datetime(timestamps)); values=np.asarray(values,np.float32)
    if values.ndim==1: values=values[:,None]
    n=max(0,len(values)-INPUT_LENGTH-FORECAST_HORIZON+1); xs=[];ys=[];origins=[];starts=[];skipped=0
    for start in range(n):
        end=start+INPUT_LENGTH+FORECAST_HORIZON
        if not np.all((time[start + 1:end] - time[start:end - 1]) == pd.Timedelta(hours=1)): skipped+=1; continue
        xs.append(values[start:start+INPUT_LENGTH]); ys.append(values[start+INPUT_LENGTH:end,0]); origins.append(time[start+INPUT_LENGTH-1]); starts.append(time[start+INPUT_LENGTH])
    return np.asarray(xs,np.float32).reshape(-1,168,values.shape[1]),np.asarray(ys,np.float32).reshape(-1,24),pd.DatetimeIndex(origins),pd.DatetimeIndex(starts),{"candidates":n,"valid":len(xs),"skipped":skipped}

def create_fold_windows(scaled_values,timestamps,validation_start,validation_end):
    """Compatibility helper: targets wholly before/within the specified outer period."""
    x,y,_,starts,_=create_continuous_sequences(scaled_values,timestamps); ends=starts+pd.Timedelta(hours=23)
    train=ends<pd.Timestamp(validation_start); valid=(starts>=pd.Timestamp(validation_start))&(ends<=pd.Timestamp(validation_end))
    return x[train],y[train],x[valid],y[valid],np.flatnonzero(valid).astype(np.int64)

def split_inner_validation(scaled_values,timestamps,outer_start):
    """Chronological split. Caller fits scaler only on values before returned inner start."""
    inner_start=pd.Timestamp(outer_start)-pd.Timedelta(hours=INNER_VALIDATION_HOURS); x,y,_,starts,stats=create_continuous_sequences(scaled_values,timestamps); ends=starts+pd.Timedelta(hours=23)
    return x[ends<inner_start],y[ends<inner_start],x[(starts>=inner_start)&(ends<pd.Timestamp(outer_start))],y[(starts>=inner_start)&(ends<pd.Timestamp(outer_start))],inner_start,stats

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train(); total=0.
    for xb,yb in loader:
        xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(); loss=criterion(model(xb),yb); loss.backward();optimizer.step();total+=loss.item()*len(xb)
    return total/len(loader.dataset)
def evaluate_loss(model,loader,criterion,device):
    model.eval();total=0.
    with torch.no_grad():
        for xb,yb in loader: total+=criterion(model(xb.to(device)),yb.to(device)).item()*len(xb)
    return total/len(loader.dataset)
def predict(model,loader,device):
    model.eval(); ps=[];ys=[]
    with torch.no_grad():
        for xb,yb in loader: ps.append(model(xb.to(device)).cpu().numpy());ys.append(yb.numpy())
    return np.vstack(ps),np.vstack(ys)
def calculate_metrics(actual,predicted):
    actual,predicted=np.asarray(actual,float).ravel(),np.asarray(predicted,float).ravel(); denominator=np.maximum(np.abs(actual),1e-8)
    return {"mae":float(mean_absolute_error(actual,predicted)),"rmse":float(np.sqrt(mean_squared_error(actual,predicted))),"mape":float(np.mean(np.abs(actual-predicted)/denominator)*100),"r2":float(r2_score(actual,predicted)) if len(actual)>1 and np.ptp(actual)>0 else float("nan")}
def calculate_horizon_metrics(actual,predicted): return pd.DataFrame([{"horizon":h+1,**calculate_metrics(actual[:,h],predicted[:,h]),"n_samples":len(actual)} for h in range(24)])
def _loader(x,y): return DataLoader(LoadForecastDataset(x,y),batch_size=BATCH_SIZE,shuffle=False)
def _inverse(scaler,a): return scaler.inverse_transform(a.reshape(-1,1)).reshape(a.shape)

def main():
    set_seed();OUTPUT_DIR.mkdir(parents=True,exist_ok=True);CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True);data=load_data();validate_folds(data.timestamp.max())
    data=data[data.timestamp<FINAL_TEST_START].reset_index(drop=True)
    assert not (data.timestamp>=FINAL_TEST_START).any(),"June 2026 must not enter CV"; assert len(FOLDS)==4
    values=data[["demand_mw"]].to_numpy(np.float32);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");print(f"Device: {device}")
    fold_rows=[];prediction_rows=[];horizon_frames=[]
    for fold,outer_start,outer_end in FOLDS:
        assert outer_end<FINAL_TEST_START,f"{fold} overlaps locked June"; inner_start=outer_start-pd.Timedelta(hours=INNER_VALIDATION_HOURS); scaler=StandardScaler().fit(values[(data.timestamp<inner_start).to_numpy()]); scaled=scaler.transform(values).astype(np.float32)
        xt,yt,xi,yi,_,stats=split_inner_validation(scaled,data.timestamp,outer_start); xa,ya,origins,starts,_=create_continuous_sequences(scaled,data.timestamp); ends=starts+pd.Timedelta(hours=23); outer=(starts>=outer_start)&(ends<=outer_end); xo,yo=xa[outer],ya[outer]
        if not all(map(len,(xt,xi,xo))): raise RuntimeError(f"{fold} has zero train, inner, or outer sequences.")
        assert xt.shape[1:]==(168,1) and yt.shape[1]==24 and not (starts[outer]>=FINAL_TEST_START).any()
        print(f"\n{fold}: train < {inner_start}; inner {inner_start} to {outer_start}; outer {outer_start} to {outer_end}; windows candidate/valid/skipped={stats['candidates']}/{stats['valid']}/{stats['skipped']}; sequences={len(xt)}/{len(xi)}/{len(xo)}; scaler=train-only")
        model=BaselineLSTM().to(device);opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE);criterion=nn.MSELoss();best_loss=float("inf");best=None;wait=0;best_epoch=0
        for epoch in range(1,EPOCHS+1):
            tl=train_one_epoch(model,_loader(xt,yt),criterion,opt,device);il=evaluate_loss(model,_loader(xi,yi),criterion,device)
            if il<best_loss: best_loss=il;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0;best_epoch=epoch
            else: wait+=1
            print(f"epoch={epoch:02d} train_loss={tl:.6f} inner_loss={il:.6f} patience={wait}/{PATIENCE}")
            if wait>=PATIENCE: break
        model.load_state_dict(best);torch.save({"model_state_dict":best,"fold":fold,"best_epoch":best_epoch,"selected_on":"inner_validation_only"},CHECKPOINT_DIR/f"fold_{fold}.pt")
        pred_scaled,act_scaled=predict(model,_loader(xo,yo),device); pred,act=_inverse(scaler,pred_scaled),_inverse(scaler,act_scaled)
        metric=calculate_metrics(act,pred);fold_rows.append({"fold":fold,"validation_start":str(outer_start),"validation_end":str(outer_end),**metric,"n_predictions":int(act.size)})
        hz=calculate_horizon_metrics(act,pred);hz.insert(0,"fold",fold);horizon_frames.append(hz)
        for i,origin in enumerate(origins[outer]):
            for h in range(24): prediction_rows.append({"fold":fold,"forecast_origin":origin,"target_timestamp":starts[outer][i]+pd.Timedelta(hours=h),"horizon":h+1,"actual_mw":float(act[i,h]),"predicted_mw":float(pred[i,h])})
        print(f"outer MAE={metric['mae']:.4f} RMSE={metric['rmse']:.4f} MAPE={metric['mape']:.4f}% R2={metric['r2']:.4f}; best epoch={best_epoch}")
    folds=pd.DataFrame(fold_rows);folds.to_csv(FOLD_METRICS_PATH,index=False);allp=pd.DataFrame(prediction_rows).sort_values(["fold","forecast_origin","horizon"]);allp.to_csv(ALL_HORIZONS_PATH,index=False);allp[allp.horizon==1].to_csv(PREDICTIONS_PATH,index=False)
    hz=pd.concat(horizon_frames);hz.groupby("horizon",as_index=False).agg(mae=("mae","mean"),rmse=("rmse","mean"),mape=("mape","mean"),r2=("r2","mean"),n_samples=("n_samples","sum")).to_csv(HORIZON_METRICS_PATH,index=False)
    summary={"model":"LSTM","context_hours":168,"forecast_horizon":24,"cv_type":"expanding_window_4_fold","folds":4,"seed":SEED}
    for n in ("mae","rmse","mape","r2"): summary[f"mean_{n}"]=float(folds[n].mean());summary[f"std_{n}"]=float(folds[n].std(ddof=0))
    METRICS_PATH.write_text(json.dumps(summary,indent=2),encoding="utf-8");print(f"Saved CV artifacts to {OUTPUT_DIR}; checkpoints are CV-only, not final models.")
def cli():
    try: main()
    except KeyboardInterrupt: print("\nTraining interrupted.");raise SystemExit(130) from None
if __name__=="__main__":cli()
