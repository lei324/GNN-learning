# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    base.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/21 20:05:23 by lei324            #+#    #+#              #
#    Updated: 2023/12/21 20:05:23 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# 流形超类

from torch.nn import Parameter


class Manifold(object):

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """计算两点间的距离"""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """将欧几里得梯度转换为黎曼梯度"""
        raise NotImplementedError

    def proj(self, p, c):
        """将p投影到流形上"""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """将 u 投影到 p 的切线空间上"""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """u 在点 p 的指数映射"""
        raise NotImplementedError

    def expmap0(self, u, c):
        """u 在原点处的指数投影"""
        raise NotImplementedError

    def logmap(self, u, p, c):
        """u 在 p 点的对数投影"""
        raise NotImplementedError

    def logmap0(self, u, c):
        """u 在原点处的对数投影"""
        raise NotImplementedError

    def mobious_add(self, x, y, c, dim=-1):
        """两点相加"""
        raise NotImplementedError

    def mobious_matvec(self, m, x, c):
        """双曲线矩阵向量乘法"""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """在流形空间上随机初始化权重"""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """在切线空间内的向量点积"""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """u 从x到y 的平行传输"""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """u 从原点到x 的平行传输 """

        raise NotImplementedError


class ManifoldPrameter(Parameter):

    """
    黎曼空间中的参数类
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
