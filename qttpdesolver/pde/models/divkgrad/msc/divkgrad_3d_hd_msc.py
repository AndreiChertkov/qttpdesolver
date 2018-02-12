# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

def k_func(x, e1, e2, e3):
    return 2. + np.sin(2.*np.pi*x/e1)*np.sin(2.*np.pi*y/e2)*np.sin(2.*np.pi*z/e3)

class Model(object):
    name = 'divkgrad_3d_hd_msc'
    txt = 'PDE (multiscale): -div(k grad u) = -1 in X=[0, 1]^3; u_dX = 0.'
    dim = 3
    L = 1.

    Kx = Func(dim, 'kx', '2. + sin(2\pi x/e1)*sin(2\pi y/e2)*sin(2\pi z/e3)')
    Kx.set_func(k_func)

    Ky = Func(dim, 'ky', '2. + sin(2\pi x/e1)*sin(2\pi y/e2)*sin(2\pi z/e3)')
    Ky.set_func(k_func)

    Kz = Func(dim, 'kz', '2. + sin(2\pi x/e1)*sin(2\pi y/e2)*sin(2\pi z/e3)')
    Kz.set_func(k_func)

    F  = Func(dim, 'f', '-1')
    F.set_expr('-1.*np.ones(x.shape)')

    p_vals  = [1.E-2, 1.E-2, 1.E-2]
    p_names = ['e1', 'e2', 'e3']
    p_forms = ['%-10.2e', '%-10.2e', '%-10.2e']