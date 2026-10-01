"""Train a new frozen LSTM before June and evaluate June exactly once."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from models.cross_validation import FINAL_TEST_START
from models.lstm.lstm_model import (BATCH_SIZE, CHECKPOINT_DIR, EPOCHS, FORECAST_HORIZON, INNER_VALIDATION_HOURS, INPUT_LENGTH, LEARNING_RATE, LoadForecastDataset, BaselineLSTM, PATIENCE, PROJECT_ROOT, calculate_horizon_metrics, calculate_metrics, create_continuous_sequences, load_data, predict, set_seed, split_inner_validation, train_one_epoch, evaluate_loss)
from torch.utils.data import DataLoader

FINAL_DIR=PROJECT_ROOT/"artifacts"/"dnn"/"final"
FINAL_TEST_END=pd.Timestamp("2026-06-30 23:00:00")
def loader(x,y,shuffle=False):
    """Shuffle only for training, so the frozen final model is fitted the same way
    as the CV runs in lstm_model.py; June predictions must stay in window order."""
    return DataLoader(LoadForecastDataset(x,y),batch_size=BATCH_SIZE,shuffle=shuffle)
def inverse(s,a): return s.inverse_transform(a.reshape(-1,1)).reshape(a.shape)
def main():
    set_seed(); FINAL_DIR.mkdir(parents=True,exist_ok=True); data=load_data()
    if data.timestamp.max()<FINAL_TEST_END: raise ValueError("June 2026 is incomplete; locked final test cannot run.")
    # Freeze configuration from CV before this point; June is never used for selection.
    # Must track INNER_VALIDATION_HOURS: split_inner_validation() derives its own
    # boundary from that constant, so a hard-coded duration here silently disagrees
    # with the window the training split actually uses.
    inner_start=FINAL_TEST_START-pd.Timedelta(hours=INNER_VALIDATION_HOURS); values=data[["demand_mw"]].to_numpy(np.float32); scaler=StandardScaler().fit(values[(data.timestamp<inner_start).to_numpy()]);scaled=scaler.transform(values).astype(np.float32)
    xt,yt,xi,yi,_,_=split_inner_validation(scaled,data.timestamp,FINAL_TEST_START); xa,ya,origins,starts,_=create_continuous_sequences(scaled,data.timestamp);ends=starts+pd.Timedelta(hours=23); june=(starts>=FINAL_TEST_START)&(ends<=FINAL_TEST_END);xo,yo=xa[june],ya[june]
    if not all(map(len,(xt,xi,xo))): raise RuntimeError("Final training has zero train, inner-validation, or June sequences.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");print(f"Device: {device}; final train < {inner_start}, inner ends before June, June is locked test.")
    model=BaselineLSTM().to(device); opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE);criterion=nn.MSELoss();best=None;best_loss=float("inf");wait=0;best_epoch=0
    train_loader,inner_loader=loader(xt,yt,shuffle=True),loader(xi,yi)
    for epoch in range(1,EPOCHS+1):
        tl=train_one_epoch(model,train_loader,criterion,opt,device);il=evaluate_loss(model,inner_loader,criterion,device);print(f"epoch={epoch:02d} train_loss={tl:.6f} inner_loss={il:.6f} patience={wait}/{PATIENCE}")
        if il<best_loss: best_loss=il;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0;best_epoch=epoch
        else: wait+=1
        if wait>=PATIENCE: break
    model.load_state_dict(best);torch.save({"model_state_dict":best,"model":"BaselineLSTM","context_hours":INPUT_LENGTH,"forecast_horizon":FORECAST_HORIZON,"best_epoch":best_epoch,"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),"selected_on":"pre_june_inner_validation_only"},FINAL_DIR/"dnn_final_model.pt")
    ps,ys=predict(model,loader(xo,yo),device);pred,actual=inverse(scaler,ps),inverse(scaler,ys);metrics=calculate_metrics(actual,pred)
    rows=[]
    for i,origin in enumerate(origins[june]):
        for h in range(24): rows.append({"forecast_origin":origin,"target_timestamp":starts[june][i]+pd.Timedelta(hours=h),"horizon":h+1,"actual_mw":float(actual[i,h]),"predicted_mw":float(pred[i,h])})
    pd.DataFrame(rows).to_csv(FINAL_DIR/"dnn_final_june_predictions_all_horizons.csv",index=False);pd.DataFrame(rows).query("horizon == 1").to_csv(FINAL_DIR/"dnn_final_june_predictions.csv",index=False);calculate_horizon_metrics(actual,pred).to_csv(FINAL_DIR/"dnn_final_june_metrics_by_horizon.csv",index=False)
    (FINAL_DIR/"dnn_final_june_metrics.json").write_text(json.dumps({"model":"LSTM","locked_test_start":str(FINAL_TEST_START),"locked_test_end":str(FINAL_TEST_END),"best_epoch":best_epoch,**metrics},indent=2),encoding="utf-8")
    print(f"Locked June metrics saved to {FINAL_DIR}; do not use them to alter the configuration.")
if __name__=="__main__": main()
