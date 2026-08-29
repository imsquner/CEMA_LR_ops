from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

def receptive_field(kernel_size:int,dilations:list[int])->int:return 1+2*(kernel_size-1)*sum(dilations)
def count_parameters(m:nn.Module)->int:return sum(p.numel() for p in m.parameters() if p.requires_grad)

class GRURegressor(nn.Module):
    def __init__(self,input_size=5,hidden_size=32,num_layers=1,head_dropout=0.):
        super().__init__(); self.gru=nn.GRU(input_size,hidden_size,num_layers,batch_first=True,bidirectional=False,dropout=0.);self.drop=nn.Dropout(head_dropout);self.head=nn.Linear(hidden_size,1)
    def forward(self,x): out,_=self.gru(x);return self.head(self.drop(out[:,-1]))

class CausalConv(nn.Module):
    def __init__(self,inc,outc,k,d):super().__init__();self.pad=(k-1)*d;self.conv=nn.Conv1d(inc,outc,k,dilation=d)
    def forward(self,x):return self.conv(F.pad(x,(self.pad,0)))

class Block(nn.Module):
    def __init__(self,inc,c,k,d,drop):
        super().__init__();self.a=CausalConv(inc,c,k,d);self.b=CausalConv(c,c,k,d);self.drop=nn.Dropout(drop);self.skip=nn.Identity() if inc==c else nn.Conv1d(inc,c,1)
    def forward(self,x):h=self.drop(F.relu(self.a(x)));h=self.drop(F.relu(self.b(h)));return h+self.skip(x)

class TCNRegressor(nn.Module):
    def __init__(self,input_size=5,channels=16,residual_blocks=2,kernel_size=3,dropout=0.,lookback=12):
        super().__init__(); ds=[2**i for i in range(residual_blocks)]
        if receptive_field(kernel_size,ds)<lookback:raise ValueError("TCN receptive field is smaller than lookback")
        blocks=[];inc=input_size
        for d in ds:blocks.append(Block(inc,channels,kernel_size,d,dropout));inc=channels
        self.network=nn.Sequential(*blocks);self.head=nn.Linear(channels,1)
    def features(self,x):return self.network(x.transpose(1,2))
    def forward(self,x):return self.head(self.features(x)[:,:,-1])

def build_model(backbone:str,params:dict):
    if backbone=="gru":return GRURegressor(5,params["hidden_size"],params["num_layers"],params["head_dropout"])
    return TCNRegressor(5,params["channels"],params["residual_blocks"],params["kernel_size"],params["dropout"],12)

