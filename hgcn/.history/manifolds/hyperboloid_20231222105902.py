# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    hyperboloid.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/21 22:35:07 by lei324            #+#    #+#              #
#    Updated: 2023/12/21 22:35:07 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#   黎曼流形
import torch
from utils.math_utils import arcosh, cosh, sinh
from base import Manifold


class Hyperboloid(Manifold):
    """
    -x0^2+x1^2+...+=-K
    c = -1/K
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_morm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        """闵可夫斯基内积"""
        res = torch.sum(x*y, dim=-1)-2*x[..., 0]*y[..., 0]  # R+1 x R+1 ---->R

        if keepdim:
            res = res.view(res.shape + (1,))  # R+1
        return res

    def minkowski_norm(self, u, keepdim=True):
        """L2范数"""
        dot = self.minkowski_dot(u, u, keepdim=keepdim)

        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1./c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod/K, min=1.0+self.eps[x.dtype])
        sqdist = K * arcosh(theta)**2
        # avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1./c
        d = x.size(-1)-1  # 双曲空间的维度比 x 的维度少一个
        # ，从最后一个维度的第二个元素开始，提取 d 个元素。这相当于去除 x 中的第一个元素（通常在双曲空间中表示时间维度）
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True)**2
        mask = torch.ones_like(x)
        mask[:, 0] = 0  # 用于在后续计算中保留除第0维的所有元素
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(
            K+y_sqnorm, min=self.eps[x.dtype]))  # 映射后的第一维元素
        return vals + mask * x
