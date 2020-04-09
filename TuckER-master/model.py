import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        if kwargs["bk"]: self.constraints=torch.tensor([1,1,0,-1,-1,-1,-1,-1,1,-1,1]).to('cuda')
        else: self.constraints=torch.zeros(11,dtype=torch.long,device='cuda')
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):        
        r = self.R(r_idx)                                 #(batch, d_r)
        W_mat = torch.einsum('ijk,bi->bjk',self.W,r)    #(batch, d_e, d_e)
        W_mat = self.hidden_dropout1(W_mat)
        
        e1 = self.E(e1_idx)                               #(batch,d_e)
        x = self.bn0(e1)                                  #(batch,d_e)
        x = self.input_dropout(x)
        
        x1 = torch.einsum('bjk,bj->bk',W_mat,x)                      #(batch, d_e)
        x1 = self.bn1(x1)
        x1 = self.hidden_dropout2(x1)
        x1 = torch.einsum('bk,nk->bn',x1, self.E.weight)
        x1 = torch.einsum('bn,b->bn',x1,1-0.5*torch.abs(self.constraints[r_idx]))
        
        x2 = torch.einsum('bjk,bk->bj',W_mat,x)                     #(batch, d_e)
        x2 = self.bn1(x2)
        x2 = self.hidden_dropout2(x2)
        x2 = torch.einsum('bj,nj->bn',x2, self.E.weight)
        x2 = torch.einsum('bn,b->bn',x2,0.5*self.constraints[r_idx])
        pred = torch.sigmoid(x1+x2)
        return pred