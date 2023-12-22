# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    hyp_layers.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/22 13:48:56 by lei324            #+#    #+#              #
#    Updated: 2023/12/22 13:48:56 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    if not args.act:
        def act(x): return x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers-1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers-1))

    if args.task in ['lp', 'nc']:             # frezz
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers-1

    if args.c is None:
        # 即曲率在每一层需要更新迭代
        # 创建一个可训练的曲率列表
        curvatures = [nn.Parameter(torch.Tensor([1.]))
                      for _ in range(n_curvatures)]
    else:
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    双曲神经网络
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features,
                                out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """双曲线性层"""

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias) -> None:
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(
            self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobious_matvec(drop_weight, x, self.c)  # 双曲空间矩阵向量乘法
        res = self.manifold.proj(mv, self.c)  # 投影到对应的流形上
        if self.use_bias:
            bias = self.manifold.proj_tan0(
                self.bias.view(1, -1), self.c)  # 投影到原点的切线空间
            hyp_bias = self.manifold.expmap0(bias, self.c)  # 映射到双曲/欧式空间
            hyp_bias = self.manifold.proj(
                hyp_bias, self.c)  # 投影会产生误差，执行此步保证在流形上
            res = self.manifold.mobious_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(Module):
    """双曲空间的激活函数"""

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        # 从流形空间映射回切线空间
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        # 映射到流形  ------ 注意一般先投影到切线空间在投影回流形 而不是直接投影
        xt = self.manifold.proj_tan0(xt, self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)


class HyperbolicGraphConvolution(nn.Module):
    """ 双曲空间图卷积"""

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features,
                                out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features,
                          dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        """
        input: x,adj
        """
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypAgg(Module):
    """
    双曲聚合函数
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()

        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)  # 先映射到切线空间

        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(
                        x[i], x, c=self.c))  # 映射到特征张量所在的双曲
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                support_t = torch.sum(
                    adj_att.unsqueeze(-1)*x_local_tangent, dim=1)
                output = self.manifold.proj(
                    self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(
            self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)
