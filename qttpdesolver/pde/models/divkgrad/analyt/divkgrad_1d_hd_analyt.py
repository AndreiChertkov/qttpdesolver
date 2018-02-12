# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

def f_func(x, w1):
  s1 = np.sin(w1*x)
  c1 = np.cos(w1*x)
  return w1*w1*(1.+x)*s1 - w1*c1

class Model(object):
    name = 'divkgrad_1d_hd_analyt'
    txt = 'PDE: -div(k grad u) = f in X=[0, 1]; u_dX = 0; u is known.'
    dim = 1
    L = 1.

    Kx = Func(dim, 'kx', '1+x')
    Kx.set_expr('1.+x')

    F  = Func(dim, 'f', 'w_1^2 (1+x) sin(w_1 x) - w_1 cos(w_1 x)')
    F.set_func(f_func)

    U  = Func(dim, 'u_real', 'sin(w_1 x)')
    U.set_expr('np.sin(w1*x)')

    Ux = Func(dim, 'ux_real', 'w_1 cos(w_1 x)')
    Ux.set_expr('w1*np.cos(w1*x)')

    p_vals  = [np.pi/L * 2]
    p_names = ['w1']
    p_forms = ['%-8.4f']
