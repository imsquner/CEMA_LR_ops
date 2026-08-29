from __future__ import annotations

import hashlib, itertools, json, random

PROTOCOL = {
    "resample": {"rule":"1h","closed":"right","label":"right"},
    "ema": {"span":9,"alpha":0.2,"adjust":False,"warmup_hours":27},
    "lookback":12,"horizon":1,"dev_fraction":0.8,
    "folds":[[0,.5,.5,.6],[0,.6,.6,.7],[0,.7,.7,.8]],
    "optimizer":"AdamW","loss":"MSELoss","max_epochs":120,
    "validate_every":2,"patience_checks":15,"gradient_clip_norm":1.0,
    "amp":False,"precision":"FP32","scheduler":None,"optuna":False,
    "candidate_seed":42,"candidate_seed_training":42,
    "formal_seeds":[42,2024,2026],"max_processes":2,
}

def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def receptive_field(kernel_size:int, blocks:int)->int:
    return 1+2*(kernel_size-1)*sum(2**i for i in range(blocks))

def generate_candidate_pools(seed:int=42, backup_count:int=30)->dict:
    rng=random.Random(seed)
    gru=list(itertools.product([32,64,96],[1,2],[0.,.1,.2],[3e-4,5e-4,1e-3,2e-3],[0.,1e-5,1e-4],[32,64]))
    tcn=[]
    for b,k,c,d,lr,wd,bs in itertools.product([2,3],[2,3,5],[16,32,64],[0.,.1,.2],[3e-4,5e-4,1e-3,2e-3],[0.,1e-5,1e-4],[32,64]):
        rf=receptive_field(k,b)
        if rf>=12:tcn.append((b,k,c,d,lr,wd,bs,rf))
    rng.shuffle(gru); rng.shuffle(tcn)
    gs=[]; ts=[]
    for i,x in enumerate(gru[:5+backup_count],1):
        h,n,d,lr,wd,bs=x; gs.append({"candidate_id":f"GRU-C{i:02d}","hidden_size":h,"num_layers":n,"head_dropout":d,"learning_rate":lr,"weight_decay":wd,"batch_size":bs})
    for i,x in enumerate(tcn[:5+backup_count],1):
        b,k,c,d,lr,wd,bs,rf=x; ts.append({"candidate_id":f"TCN-C{i:02d}","residual_blocks":b,"kernel_size":k,"channels":c,"dropout":d,"learning_rate":lr,"weight_decay":wd,"batch_size":bs,"dilations":[2**j for j in range(b)],"receptive_field":rf})
    return {"seed":seed,"gru":{"primary":gs[:5],"backups":gs[5:]},"tcn":{"primary":ts[:5],"backups":ts[5:]}}

