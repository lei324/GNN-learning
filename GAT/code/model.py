import torch
import torch.nn as nn
import torch.nn.functional as F 
from layers import GraphAttentionLayer


class GAT(nn.Module):

    """
    GAT
    """
    def __init__(self,n_feature,n_hid,n_class,dropout,alpha,n_head) -> None:
        super(GAT,self).__init__()

        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(n_feature,n_hid,dropout=self.dropout,alpha=alpha,conact=True) for _ in range(n_head)]

        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

        self.out_att = GraphAttentionLayer(n_hid*n_head,n_class,dropout=dropout,alpha=alpha,conact=False)

    
    def forward(self,x,adj):
        x = F.dropout(x,self.dropout,training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions],dim=1)
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.elu(self.out_att(x,adj))

        return F.log_softmax(x,dim=1)
