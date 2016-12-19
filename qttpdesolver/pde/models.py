# -*- coding: utf-8 -*-
import numpy as np

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
        return u0(x) + e*ksi(x/e, 0.)*u0_der(x)
        
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