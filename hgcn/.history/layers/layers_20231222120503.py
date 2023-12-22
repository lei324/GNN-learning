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
        act = lambda x:x  #不激活
    else:
        act = getattr(F,args.act)
    acts = [act] * (args.num_layers-1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp','nc']:
        dims += [args.dim]
        acts +=[act]
    return dims,acts


class GraphConvolution(Module):

    def __init__(self,in_features,out_features,dropout,act,use_bias=True):
        super(GraphConvolution,self).__init__()

        self.in_features=


