# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

def f_func(x, y, w1, w2):
    xy = x*y; xy2 = xy*y
    s1 = np.sin(w1*x*x); s2 = np.sin(w2*y)
    c1 = np.cos(w1*x*x); c2 = np.cos(w2*y)
    return (4.*w1*w1*x*x + w2*w2)*(1. +xy2)*s1*s2\
          - 2.*w1*(1. + 2.*xy2)*c1*s2 - 2.*w2*xy*s1*c2

class Model(object):
    name = 'divkgrad_2d_hd_analyt'
    txt = 'PDE: -div(k grad u) = f in X=[0, 1]^2; u_dX = 0; u is known analytic solution.'
    dim = 2
    L = 1.

    Kx = Func(dim, 'kx', '1+x*y^2')
    Kx.set_expr('1. + x*y*y')

    Ky = Func(dim, 'ky', '1+x*y^2')
    Ky.set_expr('1. + x*y*y')

    F  = Func(dim, 'f', '... according to exact solution u')
    F.set_func(f_func)

    U  = Func(dim, 'u_real', 'sin(w_1 x^2) sin(w_2 y)')
    U.set_expr('np.sin(w1*x*x) * np.sin(w2*y)')

    Ux = Func(dim, 'ux_real', '2 w_1 x cos(w_1 x^2) sin(w_2 y)')
    Ux.set_expr('w1*2.*x*np.cos(w1*x*x)*np.sin(w2*y)')

    Uy = Func(dim, 'uy_real', 'w_2 sin(w_1 x^2) cos(w_2 y)')
    Uy.set_expr('w2*np.sin(w1*x*x)*np.cos(w2*y)')

    p_vals  = [np.pi/L * 2, np.pi/L * 3]
    p_names = ['w1', 'w2']
    p_forms = ['%-8.4f', '%-8.4f']
