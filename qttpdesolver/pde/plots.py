# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plot(PDE):
    _plot(PDE.u_real, PDE.u_calc, PDE.dim, txt_title='PDE solution u')
    if PDE.dim>=1:
        _plot(PDE.ux_real, PDE.ux_calc, PDE.dim, txt_title='Derivative-x of PDE solution u')
    if PDE.dim>=2:
        _plot(PDE.uy_real, PDE.uy_calc, PDE.dim, txt_title='Derivative-y of PDE solution u')
    if PDE.dim>=3:
        _plot(PDE.uz_real, PDE.uz_calc, PDE.dim, txt_title='Derivative-z of PDE solution u')
    
def _plot(ur, uc, dim, txt_title=''):
    if ur is not None and ur.isnotnone:
        ur = ur.to_np.reshape(tuple([2**(ur.d/dim)]*dim), order='F')
    if uc is not None and uc.isnotnone:
        uc = uc.to_np.reshape(tuple([2**(uc.d/dim)]*dim), order='F')
    if dim==1:
        plot_1d(ur, uc, txt_title)
    if dim==2:
        plot_2d(ur, uc, txt_title)
    if dim==3:
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
    if u_calc is not None and u_real is not None:
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