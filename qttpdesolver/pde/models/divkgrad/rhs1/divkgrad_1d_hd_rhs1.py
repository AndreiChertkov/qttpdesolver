# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

class Model(object):
    name = 'divkgrad_1d_hd_rhs1'
    txt = 'PDE: -div(k grad u) = 1 in X=[0, 1]; u_dX = 0.'
    dim = 1
    L = 1.

    Kx = Func(dim, 'kx', '1+x')
    Kx.set_expr('1.+x')

    F  = Func(dim, 'f', '1')
    F.set_expr('np.ones(x.shape)')

    p_vals  = []
    p_names = []
    p_forms = []