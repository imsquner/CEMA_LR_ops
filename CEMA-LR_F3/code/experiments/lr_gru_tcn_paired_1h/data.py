from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

SOURCE_MAP={"time":"time_h","stack_voltage":"raw_voltage_v","current":"current_a","hydrogen_outlet_flow":"h2_out_flow_l_min","air_outlet_pressure":"air_out_pressure_mbar","coolant_inlet_temp":"coolant_in_temp_c"}
FEATURE_COLUMNS=("ema_voltage_v","current_a","h2_out_flow_l_min","air_out_pressure_mbar","coolant_in_temp_c")

@dataclass
class Windows:
    features:np.ndarray; targets:np.ndarray; anchors:np.ndarray; raw_targets:np.ndarray
    timestamps:np.ndarray; origin_positions:np.ndarray; target_positions:np.ndarray
    def subset(self,idx):
        i=np.asarray(idx); return Windows(self.features[i],self.targets[i],self.anchors[i],self.raw_targets[i],self.timestamps[i],self.origin_positions[i],self.target_positions[i])

@dataclass
class Fold:
    fold:int; train_indices:np.ndarray; val_indices:np.ndarray

@dataclass
class Scalers:
    input_mean:np.ndarray; input_scale:np.ndarray; target_mean:float; target_scale:float
    def transform_x(self,x): return ((x-self.input_mean)/self.input_scale).astype(np.float32)
    def transform_y(self,y): return ((y-self.target_mean)/self.target_scale).astype(np.float32)

def discover_data(project_root:Path,dataset:str)->Path:
    found=sorted((project_root/"data"/dataset.lower()/"processed").glob(f"{dataset.upper()}_processed_*.csv"))
    if len(found)!=1: raise FileNotFoundError(f"expected one consolidated CSV for {dataset}, got {found}")
    return found[0]

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load_source(path:Path)->pd.DataFrame:
    d=pd.read_csv(path,usecols=list(SOURCE_MAP),low_memory=False).rename(columns=SOURCE_MAP)
    return d.apply(pd.to_numeric,errors="coerce").sort_values("time_h",kind="stable").reset_index(drop=True)

def causal_hourly_resample(frame:pd.DataFrame)->pd.DataFrame:
    d=frame.copy(); origin=float(d.time_h.min())
    # Numeric TimedeltaIndex gives pandas' exact closed-right, right-label semantics.
    d.index=pd.to_timedelta(d.pop("time_h")-origin,unit="h")
    out=d.resample("1h",closed="right",label="right").mean()
    out.insert(0,"time_h",origin+out.index.total_seconds()/3600)
    return out.reset_index(drop=True)

def prepare_hourly(source:pd.DataFrame,warmup_hours:int=27)->pd.DataFrame:
    d=causal_hourly_resample(source)
    d["ema_voltage_v"]=d.raw_voltage_v.ewm(span=9,adjust=False).mean()
    return d.iloc[warmup_hours:].reset_index(drop=True)

def build_windows(frame:pd.DataFrame,lookback:int=12,horizon:int=1):
    xs=[]; ys=[]; anchors=[]; raws=[]; stamps=[]; origins=[]; targets=[]
    counts={"candidate_samples":0,"input_nan":0,"label_nan":0,"anchor_nan":0}
    vals=frame.loc[:,FEATURE_COLUMNS].to_numpy(float); y=frame.ema_voltage_v.to_numpy(float); raw=frame.raw_voltage_v.to_numpy(float); time=frame.time_h.to_numpy(float)
    for target in range(lookback-1+horizon,len(frame)):
        counts["candidate_samples"]+=1; origin=target-horizon; x=vals[origin-lookback+1:origin+1]
        if not np.isfinite(x).all(): counts["input_nan"]+=1; continue
        if not np.isfinite(y[target]): counts["label_nan"]+=1; continue
        if not np.isfinite(y[origin]): counts["anchor_nan"]+=1; continue
        xs.append(x);ys.append([y[target]]);anchors.append([y[origin]]);raws.append(raw[target]);stamps.append(time[target]);origins.append(origin);targets.append(target)
    if not xs: raise ValueError("no valid windows")
    w=Windows(np.asarray(xs,np.float32),np.asarray(ys,np.float32),np.asarray(anchors,np.float32),np.asarray(raws,float),np.asarray(stamps,float),np.asarray(origins),np.asarray(targets))
    counts["valid_samples"]=len(xs); counts["deleted_total"]=counts["candidate_samples"]-len(xs)
    return w,counts

def make_folds(n:int):
    bounds=[int(n*x) for x in (.5,.6,.7,.8)]
    folds=[Fold(i+1,np.arange(bounds[i]),np.arange(bounds[i],bounds[i+1])) for i in range(3)]
    return folds,bounds[-1]

def fit_scalers(w:Windows,indices,target_mode:str)->Scalers:
    idx=np.asarray(indices); flat=w.features[idx].reshape(-1,w.features.shape[-1]); mean=flat.mean(0); scale=flat.std(0); scale=np.where(scale==0,1,scale)
    target=w.targets[idx] if target_mode=="direct" else w.targets[idx]-w.anchors[idx]
    tm=float(target.mean()); ts=float(target.std()) or 1.0
    return Scalers(mean,scale,tm,ts)

