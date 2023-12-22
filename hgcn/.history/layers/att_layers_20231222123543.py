# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    att_layers.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/22 12:31:52 by lei324            #+#    #+#              #
#    Updated: 2023/12/22 12:31:52 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
