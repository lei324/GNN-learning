# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    math_utils.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/20 10:01:03 by lei324            #+#    #+#              #
#    Updated: 2023/12/20 10:01:03 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import torch

def cosh(x,clamp=15):
    return x.clamp(-clamp,clamp).cosh()

def sinh(x,clamp):
    return x.clamp(-clamp,clamp).sinh()

def tanh(x,clamp):
    return x.clamp(-clamp,clamp).tanh()

def arcosh(x):
    return Arcosh.apply(x)

def arsinh(x):
    return Arsinh.apply(x)

def artanh(x):
    return Artanh.apply(x)


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):         
        #ctx 是一个上下文对象，用于存储有关前向传播和反向传播计算的信息
        x = 