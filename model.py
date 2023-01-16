import numpy as np
import torch as th
from torch import nn


class EMA():
    def __init__(self,model,decay):
        self.model=model
        self.decay=decay
        self.shadow={}
        self.backup={}
    
    def register(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name]=param.data.clone()
    
    def update(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average=(1.0-self.decay)*param.data+self.decay*self.shadow[name]
                self.shadow[name]=new_average.clone()
    
    def apply_shadow(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name]=param.data
                param.data=self.shadow[name]
    
    def restore(self):
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data=self.backup[name]
        self.backup={}
        

class DinAttnModel(nn.Module):
    def __init__(self,dropout,in_dim,dim_emb_id,dim_lstm_hid,seq_len):
        super(DinAttnModel,self).__init__()
    
        self.emb_id=nn.Embedding(1605,dim_emb_id)
        self.mlp1=nn.Linear(in_dim,32,bias=False)
        self.mlp2=nn.Linear(10*(seq_len-1)+dim_lstm_hid,10,bias=False)
        self.fc=nn.Linear(10,1,bias=False)
        self.act1=nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout[0]),
        )
        self.act2=nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout[0]),
        )
        self.act3=nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout[0]),
        )
        self.lstm=nn.LSTM(
            input_size=32,
            hidden_size=dim_lstm_hid,
            batch_first=True,
            num_layers=1,
            dropout=dropout[1],
        )
        self.gin_attn=nn.ModuleList([])
        for _ in range(seq_len-1):
            self.gin_attn.append(nn.Sequential(
                nn.PReLU(),
                nn.Linear(4*dim_lstm_hid,10,bias=False),
            ))
        
    def forward(self,src,stock_id,extra_output=False):
        self.lstm.flatten_parameters()
        
#         x_emb=self.emb_id(stock_id.to(dtype=th.long))
        x=self.mlp1(src) # N*seq_len*64
        x=self.act1(x)
        x,_=self.lstm(x) # N*seq_len*32
        x=self.act2(x)
        
        seq_len=x.shape[1]
        res=th.zeros(x.shape[0],10*(seq_len-1)).to(device="cuda")
        v1=x[:,-1]
        for i in range(seq_len-1):
            v2=x[:,i]
            concat_v=th.concat((v1,v2,v1-v2,v1*v2),dim=-1)
            concat_v=self.gin_attn[i](concat_v)
            res[:,10*i:10*(i+1)]=concat_v
        
        x=th.cat((res,v1),dim=-1)
        x=self.mlp2(x) 
        x=self.act3(x)
        extra_out=x
        x=self.fc(x)
        if extra_output:
            return x,extra_out
        else:
            return x
        
        
class SelfAttnModel(nn.Module):
    def __init__(self,dropout,in_dim,dim_emb_id,dim_lstm_hid):
        super(SelfAttnModel,self).__init__()
        
        self.emb_id=nn.Embedding(1605,dim_emb_id)
        self.mlp1=nn.Linear(in_dim,dim_lstm_hid,bias=False)
        self.mlp2=nn.Linear(dim_lstm_hid*(seq_len+1),dim_lstm_hid,bias=False)
        self.mlp3=nn.Linear(dim_lstm_hid,10,bias=False)
        self.fc=nn.Linear(10,1,bias=False)
        self.act1=nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout[0]),
        )
        self.act2=nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout[0]),
        )
        self.act3=nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout[0]),
        )
        self.lstm=nn.LSTM(
            input_size=dim_lstm_hid,
            hidden_size=dim_lstm_hid,
            batch_first=True,
            num_layers=1,
            dropout=dropout[1],
        )
        
        self.self_attn=nn.MultiheadAttention(embed_dim=dim_lstm_hid,num_heads=4,batch_first=True,dropout=dropout[2])
        
    def forward(self,src,stock_id,extra_output=False):
        self.lstm.flatten_parameters()
        
#         x_emb=self.emb_id(stock_id.to(dtype=th.long))
        x=self.mlp1(src) # N*seq_len*64
        x=self.act1(x)
        x,_=self.lstm(x) # N*seq_len*32
        
        x2=x[:,-1]
        x,_=self.self_attn(x,x,x) # self_attn
        x=x.contiguous().view(x.shape[0],-1)
        x=th.concat((x,x2),dim=-1)
        x=self.mlp2(x) 
        x=self.act2(x)
#         x=th.concat((x,x_emb),dim=-1)
        extra_out=x
        x=self.mlp3(x) 
        x=self.act3(x)
        x=self.fc(x)
        if extra_output:
            return x,extra_out
        else:
            return x

        
class Stage2Model(nn.Module):
    def __init__(self,dropout,in_dim):
        super(Stage2Model,self).__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_dim,32,bias=False),
            nn.PReLU(),
            nn.Dropout(dropout[0]),
            nn.Linear(32,1,bias=False),
        )
    def forward(self,src):
        x=self.mlp(src)
        return x