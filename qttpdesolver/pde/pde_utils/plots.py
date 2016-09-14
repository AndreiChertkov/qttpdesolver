# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from ...utils.general import MODE_TT

def plot(PDE):
    plot_u(PDE)
    if PDE.dim>=1:
        plot_u_der(PDE, 0, 'Derivative-x of PDE solution u')
    if PDE.dim>=2:
        plot_u_der(PDE, 1, 'Derivative-y of PDE solution u')
    if PDE.dim>=3:
        plot_u_der(PDE, 2, 'Derivative-z of PDE solution u')
        
def plot_u(PDE):
    _plot(PDE.n, PDE.dim, PDE.u_real, PDE.u_calc, PDE.mode, 'PDE solution u')
    
def plot_u_der(PDE, num, name):
    if isinstance(PDE.ud_real, list) and len(PDE.ud_real)>num:
        ud_real = PDE.ud_real[num]
    else:
        ud_real = None
    if isinstance(PDE.ud_calc, list) and len(PDE.ud_calc)>num:
        ud_calc = PDE.ud_calc[num]
    else:
        ud_calc = None
    _plot(PDE.n, PDE.dim, ud_real, ud_calc, PDE.mode, name)
    
def _plot(n, dim, u_real, u_calc, mode, txt_title=''):
    if u_real is not None:
        if mode != MODE_TT:
            ur = u_real.copy()
        else:
            ur = u_real.full().flatten('F')
    else:  
        ur = None
        
    if u_calc is not None:
        if mode != MODE_TT:
            uc = u_calc.copy()
        else:
            uc = u_calc.full().flatten('F')
    else:  
        uc = None
           
    if dim==1:
        plot_1d(ur, uc, txt_title)
    if dim==2:
        if ur is not None:
            ur = ur.reshape((n, n), order='F')
        if uc is not None:
            uc = uc.reshape((n, n), order='F')
        plot_2d(ur, uc, txt_title)
    if dim==3:
        if ur is not None:
            ur = ur.reshape((n, n, n), order='F')
        if u_calc is not None:
            uc = uc.reshape((n, n, n), order='F')
        plot_3d(ur, uc, txt_title)
        
def plot_1d(u_real=None, u_calc=None, txt_title='PDE solution'):
    if u_real is None and u_calc is None:
        return
    plt.title(txt_title)
    if u_calc is not None:
        plt.plot(u_calc, label='calc', color='green')
    if u_real is not None:
        plt.plot(u_real, '--', label='real', color='blue')
    plt.legend(loc="best"); plt.show()

def plot_2d(u_real=None, u_calc=None, txt_title='PDE solution'):
    if u_calc is None and u_real is None:
        return
    elif u_calc is not None and u_real is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4)) 
    elif u_calc is not None and u_real is None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))  
    elif u_calc is None and u_real is not None:
        fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4)) 
    else:
        return
        
    vmin = 1.E16
    vmax =-1.E16
    if u_calc is not None:
        vmin = np.min([vmin, np.min(u_calc)])
        vmax = np.max([vmax, np.max(u_calc)])
    if u_real is not None:
        vmin = np.min([vmin, np.min(u_real)])
        vmax = np.max([vmax, np.max(u_real)])

    if u_calc is not None:
        ax1.plot(u_calc.flatten(), label='calc', color='green')    
        im = ax2.contourf(u_calc, vmin=vmin, vmax=vmax)
        ax2.set_title('Calc')
        
    if u_real is not None:
        ax1.plot(u_real.flatten(), '--', label='real', color='blue')  
        im = ax3.contourf(u_real, vmin=vmin, vmax=vmax)
        ax3.set_title('Real')
        
    ax1.set_title(txt_title)
    ax1.set_xlabel('Number of mesh point')
    ax1.legend(loc='best')
    
    cax = fig.add_axes([0.95, 0.55, 0.02, 0.35])
    fig.colorbar(im, cax=cax) 
    plt.show()
    
def plot_3d(u_real=None, u_calc=None, txt_title='PDE solution'):
    if u_real is None and u_calc is None:
        return
    plt.title(txt_title)
    if u_calc is not None:
        plt.plot(u_calc.flatten(), label='calc', color='green')
    if u_real is not None:
        plt.plot(u_real.flatten(), '--', label='real', color='blue')
    plt.legend(loc="best"); plt.show()