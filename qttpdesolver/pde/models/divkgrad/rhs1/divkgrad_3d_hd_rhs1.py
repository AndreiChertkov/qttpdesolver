# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

class Model(object):
    name = 'divkgrad_3d_hd_rhs1'
    txt = 'PDE: -div(k grad u) = 1 in X=[0, 1]^3; u_dX = 0.'
    dim = 3
    L = 1.

    Kx = Func(dim, 'kx', '1+x^2*y*z^3')
    Kx.set_expr('1. + x*x*y*z*z*z')

    Ky = Func(dim, 'ky', '1+x^2*y*z^3')
    Ky.set_expr('1. + x*x*y*z*z*z')

    Kz = Func(dim, 'kz', '1+x^2*y*z^3')
    Kz.set_expr('1. + x*x*y*z*z*z')

    F  = Func(dim, 'f', '1')
    F.set_expr('np.ones(x.shape)')

    p_vals  = []
    p_names = []
    p_forms = []