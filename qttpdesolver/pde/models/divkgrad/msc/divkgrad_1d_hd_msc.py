# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

def k_func(x, e):

    def k0(x):
        return 1.+x

    def k1(x1):
        c = np.cos(2.*np.pi*x1)
        return 2./3.*(1.+c*c)

    return k0(x)*k1(x/e)

class Model(object):
    name = 'divkgrad_1d_hd_msc'
    txt = 'PDE (multiscale): -div(k grad u) = -1 in X=[0, 1]; u_dX = 0.'
    dim = 1
    L = 1.

    Kx = Func(dim, 'kx', '2/3 (1+x) (1 + cos^2 (2 \pi x/e) )')
    Kx.set_func(k_func)

    F  = Func(dim, 'f', '-1')
    F.set_expr('-1.*np.ones(x.shape)')

    p_vals  = [1.E-4]
    p_names = ['e']
    p_forms = ['%-10.2e']