# -*- coding: utf-8 -*-
import numpy as np
from qttpdesolver import Func

def f_func(x, y, z, w1, w2, w3):
    x2 = x*x; y2 = y*y; z2 = z*z
    t = x2*y*z2; tz = t*z; q = tz+1.
    s1 = np.sin(w1*x2*x); s2 = np.sin(w2*y2); s3 = np.sin(w3*z)
    c1 = np.cos(w1*x2*x); c2 = np.cos(w2*y2); c3 = np.cos(w3*z)
    f = 9.*w1*w1*x2*x2
    f+= 4.*w2*w2*y2
    f+=    w3*w3
    f*= q*s1*s2*s3
    f-= 6.*w1*x*(1.+2.*tz)*c1*s2*s3
    f-= 2.*w2*(1.+2.*tz)*s1*c2*s3
    f-= 3.*w3*t*s1*s2*c3
    return f

class Model(object):
    name = 'divkgrad_3d_hd_analyt'
    txt = 'PDE: -div(k grad u) = f in X=[0, 1]^3; u_dX = 0; u is known analytic solution.'
    dim = 3
    L = 1.

    Kx = Func(dim, 'kx', '1+x^2*y*z^3')
    Kx.set_expr('1.+x*x*y*z*z*z')

    Ky = Func(dim, 'ky', '1+x^2*y*z^3')
    Ky.set_expr('1.+x*x*y*z*z*z')

    Kz = Func(dim, 'kz', '1+x^2*y*z^3')
    Kz.set_expr('1.+x*x*y*z*z*z')

    F  = Func(dim, 'f', '... according to exact solution u')
    F.set_func(f_func)
    
    U  = Func(dim, 'u_real', 'sin(w_1 x^3) sin(w_2 y^2) sin(w_3 z)')
    U.set_expr('np.sin(w1*x*x*x)*np.sin(w2*y*y)*np.sin(w3*z)')

    Ux = Func(dim, 'ux_real', '3 w_1 x^2 cos(w_1 x^3) sin(w_2 y^2) sin(w_3 z)')
    Ux.set_expr('w1*3.*x*x*np.cos(w1*x*x*x)*np.sin(w2*y*y)*np.sin(w3*z)') 

    Uy = Func(dim, 'uy_real', '2 w_2 y sin(w_1 x^3) cos(w_2 y^2) sin(w_3 z)')
    Uy.set_expr('w2*y*2.*np.sin(w1*x*x*x)*np.cos(w2*y*y)*np.sin(w3*z)') 

    Uz = Func(dim, 'uz_real', '2 w_2 y sin(w_1 x^3) cos(w_2 y^2) sin(w_3 z)')
    Uz.set_expr('w3*np.sin(w1*x*x*x)*np.sin(w2*y*y)*np.cos(w3*z)') 

    p_vals  = [np.pi/L * 2, np.pi/L * 3, np.pi/L * 4]
    p_names = ['w1', 'w2', 'w3']
    p_forms = ['%-8.4f', '%-8.4f', '%-8.4f']