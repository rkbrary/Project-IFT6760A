import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()
        
        self.constraints=torch.tensor([0,1,-1,0,-1,-1,-1,-1,1,0,1], device='cuda')
        self.rel_perm = torch.tensor([8, 0, 3, 9, 4, 5, 6, 7, 1, 10, 2],device='cuda')
        self.num_sym = torch.sum(self.constraints==1)
        self.num_asym = torch.sum(self.constraints==-1)
        self.d1 = d1
        self.d2 = d2
        
        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R1 = torch.nn.Embedding(int(self.num_sym), d2, padding_idx=0)
        self.R2 = torch.nn.Embedding(int(self.num_asym), d2, padding_idx=0)
        self.R3 = torch.nn.Embedding(len(d.relations)-int(self.num_sym+self.num_asym), 2*d2, padding_idx=0)
        self.W1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1*(d1+1)//2)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.W2 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1*(d1-1)//2)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        
    def mat_to_sym(self, mat, n):
        temp = torch.cat([torch.cat([torch.zeros((mat.size(0),i)), mat[:,(2*n-i+1)*i//2:(2*n-i)*(i+1)//2]],dim=1)[:,:,None] for i in range(n)],dim=2)
        return temp+temp.transpose(1,2)
    
    def mat_to_asym(self, mat, n):
        if mat.size(-1)!=n*(n-1)/2: raise Exception('Invalid dimensions')
        d2=mat.size(0)
        asym=torch.zeros((d2,n,n), device='cuda')
        pos=0
        for i in range(n):
            temp=mat[:, pos:pos+(n-i-1)]
            asym[:,i,i+1:] = temp
            asym[:,i+1:,i] = -temp
            pos+=n-i-1
        return asym
    
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R1.weight.data)
        xavier_normal_(self.R2.weight.data)
        xavier_normal_(self.R3.weight.data)
        print('new model')

    def forward(self, e1_idx, r_idx):
        R = torch.cat(
            (
            torch.cat((self.R1.weight, torch.zeros((self.num_sym, self.d2), device='cuda')), dim=1),
            torch.cat((torch.zeros((self.num_asym, self.d2), device='cuda'), self.R2.weight), dim=1),
            self.R3.weight
            ), dim=0)

        r = R[self.rel_perm[r_idx]]                                 #(batch, d_r)
        W = torch.cat(
            (
                self.mat_to_sym(self.W1, self.d1),
                self.mat_to_asym(self.W2, self.d1)
            ),dim=0)
        W_mat = torch.einsum('ijk,bi->bjk',W,r)    #(batch, d_e, d_e)
        W_mat = self.hidden_dropout1(W_mat)

        e1 = self.E(e1_idx)                               #(batch,d_e)
        x = self.bn0(e1)                                  #(batch,d_e)
        x = self.input_dropout(x)
        
        x = torch.einsum('bjk,bj->bk',W_mat,x)                      #(batch, d_e)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.einsum('bk,nk->bn',x, self.E.weight)
        
        pred = torch.sigmoid(x)
            
        return pred