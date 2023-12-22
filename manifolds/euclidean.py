# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    euclidean.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/21 21:53:01 by lei324            #+#    #+#              #
#    Updated: 2023/12/21 21:53:01 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#              欧式空间的运算

from manifolds.base import Manifold


class Euclidean(Manifold):
    """
    欧式流形
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2, c):
        return (p1-p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p+u

    def expmap0(self, u, c):
        return u

    def logmap(self, u, p, c):
        return p-u

    def logmap0(self, u, c):
        return u

    def mobious_add(self, x, y, c, dim=-1):
        return x+y

    def mobious_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=0.00001):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u*v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        return u

    def ptransp0(self, x, u, c):
        return x+u
