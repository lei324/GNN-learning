import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 


class GraphAttentionLayer(nn.Module):
    """
    simple GAT layer
    """

    def __init__(self,in_features,out_features,dropout,alpha,conact=True):
        super(GraphAttentionLayer,self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = conact

        self.W = nn.Parameter(torch.empty(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features,1)))
        nn.init.xavier_uniform_(self.a.data,gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,h,adj):
        Wh = torch.mm(h,self.W) # h.shape=(N,in_features) W.shape=(in_features,out_features)  Wh.shape=(N,out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)  # (N,N)
        attention = torch.where(adj>0,e,zero_vec)  #将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention,dim=1)
        attention = F.dropout(attention,self.dropout,training=self.training)

        h_prime = torch.matmul(attention,Wh)

        if self.concat:
            return F.elu(h_prime) 
        else:
            return h_prime

    

    def _prepare_attentional_mechanism_input(self,Wh):
        """
        Wh.shape=(N,out_features)
        self.a.shape=(2*out_features,1)
        Wh1.shape (N, 1)
        Wh2.shape(N,1)
        e.shape==(N,N)
        """
        Wh1 = torch.matmul(Wh,self.a[:self.out_features,:])  #(N,1)
        Wh2 = torch.matmul(Wh,self.a[self.out_features:,:]) #(N,1)

        e = Wh1 + Wh2.T 
        return self.leakyrelu(e)
    
    def __repr__(self):
        return self.__class__.__name__ + '('+str(self.in_features) + '->'+str(self.out_features)+')'

