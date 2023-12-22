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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from att_layers import DenseAtt


def get_dim_act(args):
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
