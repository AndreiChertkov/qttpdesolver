# -*- coding: utf-8 -*-
import numpy as np

from qttpdesolver import BC_HD, BC_PR

models_txt, models_names = [], []

def get_models_txt():
    s = ''
    for i, m in enumerate(models_txt):
        s+= 'Model #%2d. Name: |%-s|\n   %-s\n'%(i, models_names[i], m)
    return s
        
def set_model(PDE, model_selected):
    if isinstance(model_selected, int):
        model_num = model_selected
        if model_num < 0 or model_num >= len(models_txt):
            raise ValueError('Incorrect model number.')
    else:
        try:
            model_num = models_names.index(model_selected)
        except:
            raise ValueError('Incorrect model name.') 
    eval('set_model_%d'%model_num)(PDE)
    return model_num

models_txt.append('-div(k grad u) = f in [0, 1]; u_d = 0; u is known')
models_names.append('Simple. Analyt 1D diffusion PDE')
def set_model_0(PDE):
    PDE.txt = models_txt[0]
    PDE.dim = 1; PDE.L = 1.
    
    def k_func(x , w1):
        return 1. + x

    PDE.k_txt = 'k  = 1+x'
    PDE.k = k_func

    def f_func(x, w1):
        s1 = np.sin(w1*x)
        c1 = np.cos(w1*x)
        return w1*w1*(1.+x)*s1 - w1*c1

    PDE.f_txt = 'f  = w_1^2 (1+x) sin(w_1 x) - w_1 cos(w_1 x)'
    PDE.f = f_func

    def u_func(x, w1):
        return np.sin(w1*x)

    PDE.u_txt = 'u  = sin(w_1 x)'
    PDE.u = u_func

    def ux_func(x, w1):
        return w1*np.cos(w1*x)
        
    PDE.ux_txt = 'ux = w_1 cos(w_1 x)'
    PDE.ux = ux_func
    
    PDE.params = [np.pi/PDE.L * 2]
    PDE.params_txt = 'w1 [=%-8.4f]'
    
models_txt.append('-div(k grad u) = f in [0, 1]^2; u_d = 0; u is known')  
models_names.append('Simple. Analyt 2D diffusion PDE')   
def set_model_1(PDE):
    PDE.txt = models_txt[1]
    PDE.dim = 2; PDE.L = 1.
    
    def k_func(x, y, w1, w2):
        return 1. + x*y*y

    PDE.k_txt = 'k  = 1+x*y^2'
    PDE.k = k_func

    def f_func(x, y, w1, w2):
        xy = x*y; xy2 = xy*y
        s1 = np.sin(w1*x*x); s2 = np.sin(w2*y)
        c1 = np.cos(w1*x*x); c2 = np.cos(w2*y)
        return (4.*w1*w1*x*x + w2*w2)*(1. +xy2)*s1*s2\
              - 2.*w1*(1. + 2.*xy2)*c1*s2 - 2.*w2*xy*s1*c2

    PDE.f_txt = 'f  = ... according to exact solution u'
    PDE.f = f_func

    def u_func(x, y, w1, w2):
        return np.sin(w1*x*x)*np.sin(w2*y)

    PDE.u_txt = 'u  = sin(w_1 x^2) sin(w_2 y)'
    PDE.u = u_func

    def ux_func(x, y, w1, w2):
        return w1*2.*x*np.cos(w1*x*x)*np.sin(w2*y)
        
    PDE.ux_txt = 'ux = 2 w_1 x cos(w_1 x^2) sin(w_2 y)'
    PDE.ux = ux_func
    
    def uy_func(x, y, w1, w2):
        return w2*np.sin(w1*x*x)*np.cos(w2*y)
        
    PDE.uy_txt = 'uy = w_2 sin(w_1 x^2) cos(w_2 y)'
    PDE.uy = uy_func
    
    PDE.params = [np.pi/PDE.L * 2, np.pi/PDE.L * 3]
    PDE.params_txt = 'w1 [=%-8.4f], w2 [=%-8.4f]'

models_txt.append('-div(k grad u) = f in [0, 1]^3; u_d = 0; u is known') 
models_names.append('Simple. Analyt 3D diffusion PDE') 
def set_model_2(PDE):
    PDE.txt = models_txt[2]
    PDE.dim = 3; PDE.L = 1.
    
    def k_func(x, y, z, w1, w2, w3):
        return 1.+x*x*y*z*z*z

    PDE.k_txt = 'k  = 1+x^2*y*z^3'
    PDE.k = k_func

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

    PDE.f_txt = 'f  = ... according to exact solution u'
    PDE.f = f_func

    def u_func(x, y, z, w1, w2, w3):
        return np.sin(w1*x*x*x)*np.sin(w2*y*y)*np.sin(w3*z)

    PDE.u_txt = 'u  = sin(w_1 x^3) sin(w_2 y^2) sin(w_3 z)'
    PDE.u = u_func

    def ux_func(x, y, z, w1, w2, w3):
        return w1*3.*x*x*np.cos(w1*x*x*x)*np.sin(w2*y*y)*np.sin(w3*z)

    PDE.ux_txt = 'ux = 3 w_1 x^2 cos(w_1 x^3) sin(w_2 y^2) sin(w_3 z)'
    PDE.ux = ux_func
    
    def uy_func(x, y, z, w1, w2, w3):
        return w2*y*2.*np.sin(w1*x*x*x)*np.cos(w2*y*y)*np.sin(w3*z)

    PDE.uy_txt = 'uy = 2 w_2 y sin(w_1 x^3) cos(w_2 y^2) sin(w_3 z)'
    PDE.uy = uy_func
    
    def uz_func(x, y, z, w1, w2, w3):
        return w3*np.sin(w1*x*x*x)*np.sin(w2*y*y)*np.cos(w3*z)

    PDE.uz_txt = 'uz = w_3 sin(w_1 x^3) sin(w_2 y^2) cos(w_3 z)'
    PDE.uz = uz_func
    
    PDE.params = [np.pi/PDE.L * 2, np.pi/PDE.L * 3, np.pi/PDE.L * 4]
    PDE.params_txt = 'w1 [=%-8.4f], w2 [=%-8.4f], w3 [=%-8.4f]' 
    
models_txt.append('-div(k grad u) = f in [0, 1]; u_d = 0; k - 1d multiscale coeff.') 
models_names.append('Simple. Msc 1D diffusion PDE')   
def set_model_3(PDE):
    PDE.txt = models_txt[3]
    PDE.dim = 1; PDE.L = 1.
    
    def k_func(x, e):

        def k0(x):
            return 1.+x
    
        def k1(x1):
            c = np.cos(2.*np.pi*x1)
            return 2./3.*(1.+c*c)
    
        return k0(x)*k1(x/e)

    PDE.k_txt = 'k  = (1+x) 2/3 (1+cos^2(2\pi x/e))'
    PDE.k = k_func

    def f_func(x, e):
        return -1.*np.ones(x.shape)

    PDE.f_txt = 'f  = -1'
    PDE.f = f_func

    def u0(x):  
        return 3./2./np.sqrt(2.) * (x - np.log(1.+x)/np.log(2.))
        
    def u0_der(x):
        return 3./2./np.sqrt(2.) * (1. - 1./(1.+x)/np.log(2.))

    def ksi(x1, C):
        return 1./2./np.pi*np.arctan( np.tan(2.*np.pi*x1)/np.sqrt(2.) )-x1+C
     
    def ksi_der(x1):
        c = np.cos(2.*np.pi*x1)
        return np.sqrt(2.)/(1.+c*c)-1.
            
    def u_func(x, e):
        # u0: exact homogenized solution
        # second order u0(x) + e*u1(x, x/e), u1(x,y) = ksi(y)*u0_der(x)   
        return u0(x) #+ e*ksi(x/e, 0.)*u0_der(x)
        
    PDE.u_txt = 'u  = ... Exact homogenized solution (is known)'
    PDE.u = u_func
    
    def ux_func(x, e):
        # du/dx
        return u0_der(x) * (1.+ksi_der(x/e))

    PDE.ux_txt = 'ux = ... From homogenized solution and 1th order appr (is known)'
    PDE.ux = ux_func

    PDE.params = [1.E-4]
    PDE.params_txt = 'e [=%-10.2e]' 
    
models_txt.append('-div(k grad u) = f in [0, 1]^2; u_d = 0; k - 2d multiscale coeff.') 
models_names.append('Simple. Msc 2D diffusion PDE')    
def set_model_4(PDE):
    PDE.txt = models_txt[4]
    PDE.dim = 2; PDE.L = 1.
    
    def k_func(x, y, e1, e2):
        return 2. + np.sin(2.*np.pi*x/e1)*np.sin(2.*np.pi*y/e2)

    PDE.k_txt = 'k  = 2. + sin(2\pi x/e1)*sin(2\pi y/e2)'
    PDE.k = k_func

    def f_func(x, y, e1, e2):
        return -1.*np.ones(x.shape)

    PDE.f_txt = 'f  = -1'
    PDE.f = f_func

    PDE.params = [1.E-2, 1.E-2]
    PDE.params_txt = 'e1 [=%-10.2e], e2 [=%-10.2e]' 
    
models_txt.append('-div(k grad u) = f in [0, 1]^3; u_d = 0; k - 3d multiscale coeff.') 
models_names.append('Simple. Msc 3D diffusion PDE')    
def set_model_5(PDE):
    PDE.txt = models_txt[5]
    PDE.dim = 3; PDE.L = 1.
    
    def k_func(x, y, z, e1, e2, e3):
        return 2. + np.sin(2.*np.pi*x/e1)*np.sin(2.*np.pi*y/e2)*np.sin(2.*np.pi*z/e3)

    PDE.k_txt = 'k  = 2. + sin(2\pi x/e1)*sin(2\pi y/e2)*sin(2\pi z/e3)'
    PDE.k = k_func

    def f_func(x, y,z,  e1, e2, e3):
        return -1.*np.ones(x.shape)

    PDE.f_txt = 'f  = -1'
    PDE.f = f_func

    PDE.params = [1.E-2, 1.E-2, 1.E-2]
    PDE.params_txt = 'e1 [=%-10.2e], e2 [=%-10.2e], e3 [=%-10.2e]'

models_txt.append('-div(k grad u) = f in [0, 1]; u_d = 0; f=1')
models_names.append('Simple. 1D diffusion PDE with rhs=1')      
def set_model_6(PDE):
    PDE.txt = models_txt[6]
    PDE.dim = 1; PDE.L = 1.
    
    def k_func(x, w1):
        return 1. + x

    PDE.k_txt = 'k  = 1+x'
    PDE.k = k_func

    def f_func(x, w1):
        return np.ones(x.shape)

    PDE.f_txt = 'f  = 1'
    PDE.f = f_func

    PDE.params = [0]
    PDE.params_txt = '[=%-8.4f]'
    
models_txt.append('-div(k grad u) = f in [0, 1]^2; u_d = 0; f=1') 
models_names.append('Simple. 2D diffusion PDE with rhs=1')        
def set_model_7(PDE):
    PDE.txt = models_txt[7]
    PDE.dim = 2; PDE.L = 1.
    
    def k_func(x, y, w1, w2):
        return 1. + x*y*y

    PDE.k_txt = 'k  = 1+x*y^2'
    PDE.k = k_func

    def f_func(x, y, w1, w2):
        return np.ones(x.shape)

    PDE.f_txt = 'f  = 1'
    PDE.f = f_func

    PDE.params = [0, 0]
    PDE.params_txt = '[=%-8.4f], [=%-8.4f]'

models_txt.append('-div(k grad u) = f in [0, 1]^3; u_d = 0; f=1')
models_names.append('Simple. 3D diffusion PDE with rhs=1')      
def set_model_8(PDE):
    PDE.txt = models_txt[8]
    PDE.dim = 3; PDE.L = 1.
    
    def k_func(x, y, z, w1, w2, w3):
        return 1.+x*x*y*z*z*z

    PDE.k_txt = 'k  = 1+x^2*y*z^3'
    PDE.k = k_func

    def f_func(x, y, z, w1, w2, w3):
        return np.ones(x.shape)

    PDE.f_txt = 'f  = 1'
    PDE.f = f_func

    PDE.params = [0, 0, 0]
    PDE.params_txt = '[=%-8.4f], [=%-8.4f], [=%-8.4f]' 
    
models_txt.append('-delta u - k^2 u = f; in [0, 1]; u(0)=1; u_x(1)=iku(1); f=0')   
models_names.append('Simple. 1D Helmholtz PDE wuth rhs=0')      
def set_model_9(PDE):
    PDE.txt = models_txt[9]
    PDE.dim = 1; PDE.L = 1.
    
    PDE.k_txt = 'k  = Is not used'
    PDE.k = None

    def f_func(x, k, ud):
        return np.zeros(x.shape)

    PDE.f_txt = 'f  = 0'
    PDE.f = f_func

    def u_func(x, k, ud):
        return np.cos(k*x)

    PDE.u_txt = 'u  = cos(k*x) (Re of e^{ikx})'
    PDE.u = u_func

    def ux_func(x, k, ud):
        return -k*np.sin(k*x)
        
    PDE.ux_txt = 'ux = - k sin(k*x) (Re of ik e^{ikx})'
    PDE.ux = None #ux_func
    
    PDE.params = [100., 1.]
    PDE.params_txt = 'k [=%-8.4f], ud [=%-8.4f]'
    
models_txt.append('-delta u - k^2 u = f; in [0, 1]; u(0)=0; u_x(1)=iku(1); f=-1')   
models_names.append('Simple. 1D Helmholtz PDE with rhs=-1')
def set_model_10(PDE):
    PDE.txt = models_txt[10]
    PDE.dim = 1; PDE.L = 1.
    
    PDE.k_txt = 'k  = Is not used'
    PDE.k = None

    def f_func(x, k, ud):
        return -np.ones(x.shape)

    PDE.f_txt = 'f  = -1'
    PDE.f = f_func

    def u_func(x, k, ud):
        return ((np.exp(1.j*k*x)-1.j*np.sin(k*x)*np.exp(1.j*k) - 1.)/k/k)

    PDE.u_txt = 'u  = (np.exp(1.j*k*x)-1.j*np.sin(k*x)*np.exp(1.j*k) - 1.)/k/k'
    PDE.u = u_func

    def ux_func(x, k, ud):
        return -k*np.sin(k*x)
        
    PDE.ux_txt = 'ux = - k sin(k*x) (Re of ik e^{ikx})'
    PDE.ux = None #ux_func
    
    PDE.params = [100., 0.]
    PDE.params_txt = 'k [=%-8.4f], ud [=%-8.4f]'
    
models_txt.append('-div(k grad u) = f in [0, 1]^2; u_d = 0; f - 4 point sources near corners')       
models_names.append('2D diffusion PDE with 4 point sources as rhs')   
def set_model_11(PDE):
    PDE.txt = models_txt[11]
    PDE.dim = 2; PDE.L = 1.
    
    def k_func(x, y, w1, w2):
        return 1. + x*y

    PDE.k_txt = 'k  = 1+x*y'
    PDE.k = k_func

    def f_func(x, y, w1, w2):
        xy = x*y; xy2 = xy*y
        s1 = np.sin(w1*x*x); s2 = np.sin(w2*y)
        c1 = np.cos(w1*x*x); c2 = np.cos(w2*y)
        return (4.*w1*w1*x*x + w2*w2)*(1. +xy2)*s1*s2\
              - 2.*w1*(1. + 2.*xy2)*c1*s2 - 2.*w2*xy*s1*c2

    PDE.f_txt = 'f  = 4 point sources near corners'
    PDE.f_func = {'func': 'delta', 
                  'r_list': [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2]],
                  'v_list': [3., 3., 3., 3.]}

    PDE.params = [np.pi/PDE.L * 2, np.pi/PDE.L * 3]
    PDE.params_txt = 'w1 [=%-8.4f], w2 [=%-8.4f]'
    

models_txt.append('-div(grad u) = f in [0, 1]; u(0) = u(1); \int_0^1 u = 0')
models_names.append('Simple. Analyt 1D periodic Poisson PDE')
def set_model_12(PDE):
    PDE.txt = models_txt[12]
    PDE.dim = 1; PDE.L = 1.; PDE.bc = BC_PR
    PDE.k_txt  = 'k  = 1'
    PDE.f_txt  = 'f  = sin(w_1 x)'
    PDE.u_txt  = 'u  = sin(w_1 x) / w1^2'
    PDE.ux_txt = 'ux = cos(w_1 x) / w1'
    
    PDE.params = [np.pi/PDE.L * 2]
    PDE.params_txt = 'w1 [=%-8.4f]'
    
    def k_func(x, w1):  return np.ones(x.shape)
    def f_func(x, w1):  return np.sin(w1*x)
    def u_func(x, w1):  return np.sin(w1*x)/w1/w1
    def ux_func(x, w1): return np.cos(w1*x)/w1

    PDE.k, PDE.f, PDE.u, PDE.ux = k_func, f_func, u_func, ux_func

models_txt.append('-div(k grad u) = f in [0, 1]; u 1 periodic; \int_0^1 u = 0')
models_names.append('Simple. Analyt 1D periodic PDE')
def set_model_13(PDE):  
    PDE.txt = models_txt[13]
    PDE.dim = 1; PDE.L = 1.; PDE.bc = BC_PR
    PDE.k_txt  = 'k  = cos(wx) + 2'
    PDE.f_txt  = 'f  = w^2 sin(wx+s)cos(wx)+w^2cos(wx+s)sin(wx)+2w^2sin(wx+s)'
    PDE.u_txt  = 'u  = sin(wx+s)'
    PDE.ux_txt = 'ux = w*cos(wx+s)'

    PDE.params = [2.*np.pi, np.pi/4.]
    PDE.params_txt = 'w [=%-8.4f], s [=%-8.4f]'

    def k_func(x, w, s):  return np.cos(w*x) + 2.
    def f_func(x, w, s):  return w*w*np.sin(w*x+s)*np.cos(w*x)+w*w*np.cos(w*x+s)*np.sin(w*x)+2.*w*w*np.sin(w*x+s)
    def u_func(x, w, s):  return np.sin(w*x+s)
    def ux_func(x, w, s): return w*np.cos(w*x+s)
    
    PDE.k, PDE.f, PDE.u, PDE.ux = k_func, f_func, u_func, ux_func
    
models_txt.append('-div(k grad u) = f in [0, 1]^2; u (1, 1) periodic; \int_\Omega u = 0')
models_names.append('Simple. Analyt 2D periodic PDE')
def set_model_14(PDE):  
    PDE.txt = models_txt[14]
    PDE.dim = 2; PDE.L = 1.; PDE.bc = BC_PR
    PDE.k_txt  = 'k  = cos(w_1 x) sin(w_2 y) + 2'
    PDE.f_txt  = 'f  = ... according to exact solution u'
    PDE.u_txt  = 'u  = sin(w_1 x+s_1) sin(w_2 y+s_2)'
    PDE.ux_txt = 'ux = w_1 cos(w_1 x+s_1) sin(w_2 y+s_2)'
    PDE.uy_txt = 'uy = w_2 sin(w_1 x+s_1) cos(w_2 y+s_2)'
    
    PDE.params = [2.*np.pi, np.pi/4., 4.*np.pi, np.pi/6.]
    PDE.params_txt = 'w1 [=%-8.4f], s1 [=%-8.4f], w2 [=%-8.4f], s2 [=%-8.4f]'

    def k_func(x, y, w1, s1, w2, s2):
        return np.cos(w1*x) * np.sin(w2*y) + 2.
    def kx_func(x, y, w1, s1, w2, s2):
        return -w1* np.sin(w1*x) * np.sin(w2*y)
    def ky_func(x, y, w1, s1, w2, s2):
        return w2* np.cos(w1*x) * np.cos(w2*y)  
    def u_func(x, y, w1, s1, w2, s2):
        return np.sin(w1*x+s1) * np.sin(w2*y+s2)
    def ux_func(x, y, w1, s1, w2, s2):
        return w1 * np.cos(w1*x+s1) * np.sin(w2*y+s2)
    def uxx_func(x, y, w1, s1, w2, s2):
        return -w1 * w1 * np.sin(w1*x+s1) * np.sin(w2*y+s2)
    def uy_func(x, y, w1, s1, w2, s2):
        return w2 * np.sin(w1*x+s1) * np.cos(w2*y+s2)
    def uyy_func(x, y, w1, s1, w2, s2):
        return -w2 * w2 * np.sin(w1*x+s1) * np.sin(w2*y+s2)
    def f_func(x, y, w1, s1, w2, s2):
        k = k_func(x, y, w1, s1, w2, s2)
        kx = kx_func(x, y, w1, s1, w2, s2)
        ky = ky_func(x, y, w1, s1, w2, s2)
        ux = ux_func(x, y, w1, s1, w2, s2)
        uy = uy_func(x, y, w1, s1, w2, s2)
        uxx = uxx_func(x, y, w1, s1, w2, s2)
        uyy = uyy_func(x, y, w1, s1, w2, s2)
        return -1. * (kx*ux+k*uxx + ky*uy + k*uyy)
        
    PDE.k, PDE.f, PDE.u, PDE.ux, PDE.uy = k_func, f_func, u_func, ux_func, uy_func
    
models_txt.append('-div(k grad u) = f in [0, 1]^3; u (1, 1, 1) periodic; \int_\Omega u = 0')
models_names.append('Simple. Analyt 3D periodic PDE')
def set_model_15(PDE):  
    PDE.txt = models_txt[15]
    PDE.dim = 3; PDE.L = 1.; PDE.bc = BC_PR
    PDE.k_txt  = 'k  = cos(w_1 x) sin(w_2 y) cos(w_3 z) + 2'
    PDE.f_txt  = 'f  = ... according to exact solution u'
    PDE.u_txt  = 'u  = sin(w_1 x+s_1) sin(w_2 y+s_2) sin(w_3 z+s_3)'
    PDE.ux_txt = 'ux = w_1 cos(w_1 x+s_1) sin(w_2 y+s_2) sin(w_3 z+s_3)'
    PDE.uy_txt = 'uy = w_2 sin(w_1 x+s_1) cos(w_2 y+s_2) sin(w_3 z+s_3)'
    PDE.uy_txt = 'uz = w_3 sin(w_1 x+s_1) sin(w_2 y+s_2) cos(w_3 z+s_3)'
    
    PDE.params = [2.*np.pi, np.pi/4., 4.*np.pi, np.pi/6., 6.*np.pi, np.pi/3.]
    PDE.params_txt = 'w1 [=%-8.4f], s1 [=%-8.4f], w2 [=%-8.4f], s2 [=%-8.4f], w3 [=%-8.4f], s3 [=%-8.4f]'

    def k_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return np.cos(w1*x) * np.sin(w2*y) * np.cos(w3*z) + 2.
    def kx_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return -w1* np.sin(w1*x) * np.sin(w2*y) * np.cos(w3*z)
    def ky_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return w2* np.cos(w1*x) * np.cos(w2*y) * np.cos(w3*z)
    def kz_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return -w3* np.cos(w1*x) * np.sin(w2*y) * np.sin(w3*z) 
    def u_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return np.sin(w1*x+s1) * np.sin(w2*y+s2) * np.sin(w3*z+s3)
    def ux_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return w1 * np.cos(w1*x+s1) * np.sin(w2*y+s2) * np.sin(w3*z+s3)
    def uxx_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return -w1 * w1 * np.sin(w1*x+s1) * np.sin(w2*y+s2) * np.sin(w3*z+s3)
    def uy_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return w2 * np.sin(w1*x+s1) * np.cos(w2*y+s2) * np.sin(w3*z+s3)
    def uyy_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return -w2 * w2 * np.sin(w1*x+s1) * np.sin(w2*y+s2) * np.sin(w3*z+s3)
    def uz_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return w3 * np.sin(w1*x+s1) * np.sin(w2*y+s2) * np.cos(w3*z+s3)
    def uzz_func(x, y, z, w1, s1, w2, s2, w3, s3):
        return -w3 * w3 * np.sin(w1*x+s1) * np.sin(w2*y+s2) * np.sin(w3*z+s3)
    def f_func(x, y, z, w1, s1, w2, s2, w3, s3):
        k = k_func(x, y, z, w1, s1, w2, s2, w3, s3)
        kx = kx_func(x, y, z, w1, s1, w2, s2, w3, s3)
        ky = ky_func(x, y, z, w1, s1, w2, s2, w3, s3)
        kz = kz_func(x, y, z, w1, s1, w2, s2, w3, s3)
        ux = ux_func(x, y, z, w1, s1, w2, s2, w3, s3)
        uy = uy_func(x, y, z, w1, s1, w2, s2, w3, s3)
        uz = uz_func(x, y, z, w1, s1, w2, s2, w3, s3)
        uxx = uxx_func(x, y, z, w1, s1, w2, s2, w3, s3)
        uyy = uyy_func(x, y, z, w1, s1, w2, s2, w3, s3)
        uzz = uzz_func(x, y, z, w1, s1, w2, s2, w3, s3)
        return -1. * (kx*ux+k*uxx + ky*uy + k*uyy + kz*uz + k*uzz)
        
    PDE.k, PDE.f, PDE.u, PDE.ux, PDE.uy, PDE.uz = k_func, f_func, u_func, ux_func, uy_func, uz_func
   
models_txt.append('-div(k grad u) = f in [0, 1]^2; u 1-periodic; k - 2d multiscale coeff.') 
models_names.append('Msc 2D diffusion periodic PDE')    
def set_model_16(PDE):
    PDE.txt = models_txt[16]
    PDE.dim = 2; PDE.L = 1.; PDE.bc = BC_PR
    
    def k_func(x, y, e1, e2):
        return 2. + np.sin(2.*np.pi*x/e1)*np.sin(2.*np.pi*y/e2)

    PDE.k_txt = 'k  = 2. + sin(2 \pi x/e1)*sin(2 \pi y/e2)'
    PDE.k = k_func

    def f_func(x, y, e1, e2):
        return -10.*np.sin(2.*np.pi*x)*np.cos(4.*np.pi*y)

    PDE.f_txt = 'f  = -10.*sin(2 \pi x)*cos(4 \pi y)'
    PDE.f = f_func

    PDE.params = [1.E-2, 1.E-2]
    PDE.params_txt = 'e1 [=%-10.2e], e2 [=%-10.2e]' 
    
#models_txt.append('-div(k grad u) = f in [0, 1]; u 1 periodic; \int_0^1 u = 0')
#models_names.append('Simple. Analyt 1D cell problem.')
#def set_model_14(PDE):  
#    PDE.txt = models_txt[14]
#    PDE.dim = 1; PDE.L = 1.
#    PDE.k_txt  = 'k  = sin(wx) + 2'
#    PDE.f_txt  = 'f  = w cos(wx) {=dk / dx}'
#    PDE.u_txt  = 'u  = ?'
#    PDE.ux_txt = 'ux = c1 / k - 1'
#
#    PDE.params = [2.*np.pi]
#    PDE.params_txt = 'w1 [=%-8.4f]'
#
#    def k_func(x, w1):  return np.sin(w1*x) + 2.
#    def f_func(x, w1):  return w1 * np.cos(w1*x)
#    def u_func(x, w1):  return np.ones(x.shape)
#    def ux_func(x, w1):
#        c1 = np.sqrt(3.) * w1 / 2.
#        c1/= np.arctan((2.*np.tan(w1/2.)+1.)/np.sqrt(3.)) - np.pi/6.
#        return c1 / k_func(x , w1) - 1.
#    
#    PDE.k, PDE.f, PDE.u, PDE.ux = k_func, f_func, u_func, ux_func






models_txt.append('-div(k grad u) = f in [0, 1]; BC: u=0; k - 1d multiscale coeff.') 
models_names.append('Analyt num hom 1D')    
def set_model_17(PDE):
    PDE.txt = models_txt[17]
    PDE.dim = 1; PDE.L = 1.; PDE.bc = BC_HD
    PDE.k_txt  = 'k  = (1+x) 2/3 (1+cos^2(2\pi x/e))'
    PDE.f_txt  = 'f  = -1'
    PDE.u_txt  = 'u  = ... Exact homogenized solution (is known)'
    PDE.ux_txt = 'ux = ... From homogenized solution and 1th order appr (is known)'

    PDE.params = [1.E-8]
    PDE.params_txt = 'e [=%-10.2e]'

    def k0_func(x, e=None):
        return x + 1.
    
    def k1_func(x, e=1):
        # x should in e-units here
        c = np.cos(2.*np.pi*x)
        return 2./3.*(1.+c*c)
    
    def k1_der_func(x, e=None):
        # x should in e-units here
        c = np.cos(2.*np.pi*x)
        s = np.sin(2.*np.pi*x)
        return -8.*np.pi/3.*c*s
    
    def k_func(x, e):
        return k0_func(x)*k1_func(x/e)
    
    def f_func(x, e=None):
        return -1.*np.ones(x.shape)
    
    def u0(x, e=None):  
        return 3./2./np.sqrt(2.) * (x - np.log(1.+x)/np.log(2.))
        
    def u0_der(x, e=None):
        return 3./2./np.sqrt(2.) * (1. - 1./(1.+x)/np.log(2.))
    
    def ksi(x, e=None, C=0.):
        # x should in e-units here
        return 1./2./np.pi*np.arctan( np.tan(2.*np.pi*x)/np.sqrt(2.) )-x+C
    
    def ksi_der(x, e=None):
        # x should in e-units here
        c = np.cos(2.*np.pi*x)
        return np.sqrt(2.)/(1.+c*c)-1.
    
    def u_func(x, e=None):
        # u0: exact homogenized solution
        # second order u0(x) + e*u1(x, x/e), u1(x,y) = ksi(y)*u0_der(x)   
        return u0(x) #+ e*ksi(x, e)*u0_der(x)
        
    def ux_func(x, e):
        # du/dx
        return u0_der(x) * (1.+ksi_der(x))
        
    PDE.k, PDE.f, PDE.u, PDE.ux = k_func, f_func, u_func, ux_func
    PDE.k0 = k0_func
    PDE.k1 = k1_func
    PDE.k1_der = k1_der_func
    
models_txt.append('-div(k grad u) = f in [0, 1]; BC: u=0; k - 1d multiscale coeff.') 
models_names.append('Analyt num hom 1D mod')    
def set_model_18(PDE):
    PDE.txt = models_txt[18]
    PDE.dim = 1; PDE.L = 1.; PDE.bc = BC_HD
    PDE.k_txt  = 'k  = (1+x) 2/3 (1+cos^2(2\pi x/e))'
    PDE.f_txt  = 'f  = -1'
    PDE.u_txt  = 'u  = ... Exact homogenized solution (is known)'
    PDE.ux_txt = 'ux = ... From homogenized solution and 1th order appr (is known)'

    PDE.params = [1.E-8]
    PDE.params_txt = 'e [=%-10.2e]'

    def k0_func(x, e=None):
        return x + 1.
    
    def k1_func(x, e=1):
        # x should in e-units here
        c = np.cos(2.*np.pi*x)
        return 2./3.*(1.+c*c)
    
    def k1_der_func(x, e=None):
        # x should in e-units here
        c = np.cos(2.*np.pi*x)
        s = np.sin(2.*np.pi*x)
        return -8.*np.pi/3.*c*s
    
    def k_func(x, e):
        return k0_func(x)*k1_func(x/e)
    
    def f_func(x, e=None):
        return -1.*np.ones(x.shape)
    
    def u0(x, e=None):  
        return 3./2./np.sqrt(2.) * (x - np.log(1.+x)/np.log(2.))
        
    def u0_der(x, e=None):
        return 3./2./np.sqrt(2.) * (1. - 1./(1.+x)/np.log(2.))
    
    def ksi(x, e=None, C=0.):
        # x should in e-units here
        return 1./2./np.pi*np.arctan( np.tan(2.*np.pi*x)/np.sqrt(2.) )-x+C
    
    def ksi_der(x, e=None):
        # x should in e-units here
        c = np.cos(2.*np.pi*x)
        return np.sqrt(2.)/(1.+c*c)-1.
    
    def rho(x):
        return np.minimum(x, 1. - x)
    
    def _sig(t):
        return t * (2. - t) if t <= 1. else 1.
    sig = np.vectorize(_sig)

    def u_func(x, e=None):
        # u0: exact homogenized solution
        # second order u0(x) + e*u1(x, x/e), u1(x,y) = ksi(y)*u0_der(x)   
        return u0(x) + e*sig(rho(x)/e)*ksi(x/e)*u0_der(x)
        
    def ux_func(x, e):
        # du/dx
        return u0_der(x) * (1.+ksi_der(x))
        
    PDE.k, PDE.f, PDE.u, PDE.ux = k_func, f_func, u_func, ux_func
    PDE.k0 = k0_func
    PDE.k1 = k1_func
    PDE.k1_der = k1_der_func
    
models_txt.append('-div(k grad u) = f in [0, 1]; u_d = 0; u is known')
models_names.append('xxx')
def set_model_19(PDE):
    PDE.txt = models_txt[19]
    PDE.dim = 1; PDE.L = 1.
    
    def k_func(x , e=1.):
        return 1. / ( 2. + np.sin(2.*np.pi*x/e) )

    PDE.k_txt = 'k  = ( 2 + sin(2\pi x/e) )^{-1}'
    PDE.k = k_func

    def f_func(x, e=None):
        return -1.*np.ones(x.shape)

    PDE.f_txt = 'f  = -1'
    PDE.f = f_func

    def u_func(x, e):
        w = 2.*np.pi/e
        s1 = np.sin(w*x)
        c1 = np.cos(w*x)
        C1 = 1. - 1./w*np.cos(w) + 1./w/w*np.sin(w)
        C1/= 1./w*np.cos(w) - 1./w - 2.
        C2 = C1/w
        return x*x + 2.*C1*x - (x+C1)/w*c1 + 1./w/w*s1 + C2

    PDE.u_txt = 'u  = exact is known'
    PDE.u = u_func

    def ux_func(x, e):
        w = 2.*np.pi/e
        s1 = np.sin(w*x)
        C1 = 1. - 1./w*np.cos(w) + 1./w/w*np.sin(w)
        C1/= 1./w*np.cos(w) - 1./w - 2.
        return (x+C1)*(2.+s1)
        
    PDE.ux_txt = 'ux = exact is known'
    PDE.ux = ux_func
    
    PDE.params = [1.E-8]
    PDE.params_txt = 'e [=%-8.2e]'
    
    
models_txt.append('-div(k grad u) = f in [0, 1]; BC: u=0; k - 1d multiscale coeff.') 
models_names.append('Analyt num hom 1D (k periodic)')    
def set_model_20(PDE):
    PDE.txt = models_txt[20]
    PDE.dim = 1; PDE.L = 1.; PDE.bc = BC_HD
    PDE.k_txt  = 'k  = (2+cos(2\pi x/e))^{-1}'
    PDE.f_txt  = 'f  = -1'
    PDE.u_txt  = 'u  = ... Exact solution is known'
    PDE.ux_txt = 'ux = ... Exact derivative is known'

    PDE.params = [1.E-8]
    PDE.params_txt = 'e [=%-10.2e]'

    def k_func(x, e=1.):
        w = 2.*np.pi/e
        c = np.cos(w*x)
        return 1./(2.+c)
    
    def f_func(x, e=None):
        return -1.*np.ones(x.shape)

    def u_func(x, e=1.):
        w = 2.*np.pi/e
        c = np.cos(w*x)
        s = np.sin(w*x)
        A =-1./2.
        B =-1./w/w
        return x*x + 2.*A*x + A/w*s + x/w*s + 1./w/w*c + B
        
    def ux_func(x, e):
        w = 2.*np.pi/e
        c = np.cos(w*x)
        A =-1./2.
        return 2.*x + 2.*A + A*c + x*c
        
    PDE.k, PDE.f, PDE.u, PDE.ux = k_func, f_func, u_func, ux_func

models_txt.append('-div(k grad u) = f in [0, 1]^2; BC: u=0; k - 2d multiscale coeff.') 
models_names.append('Analyt num hom 2D (k periodic)')    
def set_model_21(PDE):
    PDE.txt = models_txt[21]
    PDE.dim = 2; PDE.L = 1.; PDE.bc = BC_HD
    PDE.k_txt  = 'k  = (2+cos(2\pi x/e))^{-1}'
    PDE.f_txt  = 'f  = -1'
    PDE.u_txt  = 'u  = ... Exact solution is known'
    PDE.ux_txt = 'ux = ... Exact derivative is known'
    PDE.uy_txt = 'uy = ... Exact derivative is known'
    
    PDE.params = [1.E-8]
    PDE.params_txt = 'e [=%-10.2e]'

    def k_func(x, y, e=1.):
        w = 2.*np.pi/e
        c = np.cos(w*x)
        return 1./(2.+c)
    
    def f_func(x, y, e=None):
        return -1.*np.ones(x.shape)

    def u_func(x, y, e=1.):
        w = 2.*np.pi/e
        c = np.cos(w*x)
        s = np.sin(w*x)
        A =-1./2.
        B =-1./w/w
        return x*x + 2.*A*x + A/w*s + x/w*s + 1./w/w*c + B
        
    def ux_func(x, y, e):
        w = 2.*np.pi/e
        c = np.cos(w*x)
        A =-1./2.
        return 2.*x + 2.*A + A*c + x*c
        
    def uy_func(x, y, e):
        w = 2.*np.pi/e
        c = np.cos(w*x)
        A =-1./2.
        return 2.*x + 2.*A + A*c + x*c
    
    PDE.k, PDE.f = k_func, f_func#, PDE.u, PDE.ux, PDE.uy #, u_func, ux_func, uy_func
