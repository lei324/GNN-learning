# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layers.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/22 11:41:41 by lei324            #+#    #+#              #
#    Updated: 2023/12/22 11:41:41 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#  欧式空间的layer

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


def get_dim_get(args):
    """get  layers activation  and dimension """
    if not args.act:
        def act(x): return x  # 不激活
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers-1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'nc']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, dropout, act, use_bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.linear = nn.Linear(in_features, out_features, use_bias)

    def forward(self, input):
        """
        input : x,adj
        """
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        if adj.is_sparse:  # 稀疏存储
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)

        output = self.act(support), adj
        return output

    def extra_repr(self) -> str:
        return f'input_dim {self.in_features},output_dim {self.out_features}'


class FermiDiracDecoder(Module):
    """费米-狄拉克 距离 计算预测连接的概率"""

    def __init__(self, r, t) -> None:
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dict-self.r) / self.t)+1.0)
        return probs
