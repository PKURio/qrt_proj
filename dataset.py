import torch as th
from torch.utils.data import Dataset,DataLoader


class NormalDataset(Dataset):
    def __init__(self,seq_len,data,ranges,dict_colname,valid_idx=None):
        self.data=th.tensor(data,dtype=th.float32)
        (start_t,end_t)=ranges
        self.dict_colname=dict_colname
        
        cnt=0
        sample_num=self.data.shape[0]
        col_stock_id=self.dict_colname["stock_id"]
        col_trade_date=self.dict_colname["trade_date"]
        self.look_back=seq_len-1
        
        if valid_idx is None:
            self.valid_idx=th.zeros(int(5e6)).int()
            s,e=self.look_back,self.look_back
            while True:
                while s<sample_num and (self.data[s,col_stock_id]!=self.data[s-self.look_back,col_stock_id]\
                or self.data[s,col_trade_date]<start_t or self.data[s,col_trade_date]>=end_t):
                    s+=1

                if s>=sample_num:
                    break
                e=s

                while e<sample_num and self.data[e,col_stock_id]==self.data[s,col_stock_id]\
                and self.data[e,col_trade_date]>=start_t and self.data[e,col_trade_date]<end_t:
                    e+=1

                se_len=e-s
                self.valid_idx[cnt:cnt+se_len]=th.tensor(range(s,e)).int()
                cnt+=se_len
                s=e
                
            self.valid_idx=self.valid_idx[:cnt]
        else:
            self.valid_idx=valid_idx
                
        print(f"train len {self.valid_idx.shape[0]}")
    
    def __len__(self):
        return self.valid_idx.shape[0]
    
    def __getitem__(self,index):
        col_stock_id=self.dict_colname["stock_id"]
        col_return=self.dict_colname["return"]
        
        data_idx=self.valid_idx[index]
        stock_id=self.data[data_idx,col_stock_id]
        label=self.data[data_idx,col_return]
        src=self.data[data_idx-self.look_back:data_idx+1,:-3] # !!
        return src,stock_id,label

    
class PanelDataset(Dataset):
    def __init__(self,seq_len,data,ranges,dict_colname,dict_trade_date,valid_idx=None):
        self.data=th.tensor(data,dtype=th.float32)
        col_stock_id=dict_colname["stock_id"]
        col_trade_date=dict_colname["trade_date"]
        np_trade_date=data[:,col_trade_date]
        
        sample_num=self.data.shape[0]
        (start_t,end_t)=ranges
        self.dict_colname=dict_colname
        self.dict_trade_date=dict_trade_date
        self.seq_len=seq_len
        self.look_back=seq_len-1
        
        if valid_idx is None:
            self.valid_idx=th.zeros(int(3e3),1605).int() # idx 0 is valid, because seq_len > 1
            for i in range(self.look_back,sample_num):
                if np_trade_date[i]<start_t or np_trade_date[i]>=end_t\
                or self.data[i,col_stock_id]!=self.data[i-self.look_back,col_stock_id]:
                    continue
                
                d=self.dict_trade_date[np_trade_date[i]]
                sid=self.data[i,col_stock_id].to(dtype=th.int)
                self.valid_idx[d,sid]=i
        else:
            self.valid_idx=valid_idx
        
        self.start_idx=min(th.nonzero(th.sum(self.valid_idx,dim=1))).item()
        self.end_idx=max(th.nonzero(th.sum(self.valid_idx,dim=1))).item()+1
        print(f"train len {self.end_idx-self.start_idx}")
    
    def __len__(self):
        return self.end_idx-self.start_idx
    
    def __getitem__(self,index):
        panel_idx=self.start_idx+index
        src=th.zeros(1605,self.seq_len,self.data.shape[-1]-3)
        label=th.zeros(1605)
        col_return=self.dict_colname["return"]
        
        for i,data_idx in enumerate(self.valid_idx[panel_idx]):
            if data_idx == 0:
                label[i]=-1 # invalid label
            else:
                src[i]=self.data[data_idx-self.look_back:data_idx+1,:-3]
                label[i]=self.data[data_idx,col_return]

        return src,label