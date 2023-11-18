import torch
import math

from torch.nn.parameter import Parameter
from torch.nn.modules import Module


class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()

        self.in_features=in_features  #输入维度
        self.out_features=out_features #输出维度
        self.weight=Parameter(torch.FloatTensor(in_features,out_features))

        if bias:
            self.bias=Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)  #定义为可学习的参数 None

        self.reset_parameter()
        
    def reset_parameter(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def __repr__(self):
        return self.__class__.__name__+'('+str(self.in_features)+'->'+str(self.out_features)+')'
    
    def forward(self,input,adj):
        support=torch.mm(input,self.weight)
        output=torch.spmm(adj,support)
        if self.bias is not None:
            return output+self.bias
        else:
            return output
        

