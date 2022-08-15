# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

class Model(object):
    name = 'divkgrad_2d_hd_rhs1'
    txt = 'PDE: -div(k grad u) = 1 in X=[0, 1]^2; u_dX = 0.'
    dim = 2
    L = 1.

    Kx = Func(dim, 'kx', '1+x*y^2')
    Kx.set_expr('1. + x*y*y')

    Ky = Func(dim, 'ky', '1+x*y^2')
    Ky.set_expr('1. + x*y*y')

    F  = Func(dim, 'f', '1')
    F.set_expr('np.ones(x.shape)')

    p_vals  = []
    p_names = []
    p_forms = []