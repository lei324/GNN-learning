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
            K+y_sqnorm, min=self.eps[x.dtype]))  # 映射后的第一个元素
        return vals + mask * x

    def proj_tan(self, u, x, c):
        K = 1./c
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d)*u.narrow(-1, 1, d),
                       dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask*u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u-vals

    def expmap(self, u, x, c):
        """u 在切空间中的向量,x 双曲空间中的一个点"""
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta  # 在双曲空间中沿 u 方向移动 x。
        return self.proj(result, c)  # 将结果投影回双曲空间  因为指数映射的结果可能会轻微偏离双曲流形

    def expmap0(self, u, c):
        """用于在双曲空间中执行从原点出发的指数映射的 u 是切空间中的向量"""
        K = 1. / c
        sqrtK = K ** 0.5
        # 计算 u 的维度，并从 u 中提取除第一个元素外的所有元素作为 x。x 代表在除时间维度外的空间维度上的向量。
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        # 计算L2范数
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)

        # 移动  具体见论文
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)  # 计算第一个元素
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm  # 计算剩余元素
        return self.proj(res, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K,
                         max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobious_add(self, x, y, c, dim=-1):
        u = self.logmap0(y, c)  # 映射回原点切向空间
        v = self.ptransp0(x, u, c)  # u 从原点的切向空间 平行传输到 x的双曲空间
        return self.expmap(v, x, c)

    def mobious_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u@m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(
            y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        """从双曲流形 映射到庞加莱圆盘模型"""
        K = 1./c
        sqrtK = K ** 0.5
        d = x.size(-1)-1
        return sqrtK * x.narrow(-1, 1, d) / (x[:0:1] + sqrtK)
