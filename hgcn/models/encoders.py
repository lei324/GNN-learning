# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    encoder.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/22 16:36:36 by lei324            #+#    #+#              #
#    Updated: 2023/12/22 16:36:36 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
import layers.hyp_layers as hyp_layers


class Encoder(nn.Module):
    """
    Encoder abstract class
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class HGCN(Encoder):
    def __init__(self, c, args):
        super(HGCN, self).__init__(c)

        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims)-1):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim, out_dim = dims[i], dims[i+1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)
