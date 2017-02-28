# -*- coding: utf-8 -*-
import time
from numpy.linalg import norm

from solvers.solver import create_solver
from tensor_wrapper import Vector

def auto_solve(PDE, PDESolver=None, present_res_1s=True, return_solver=False, 
               d=None, apply2pde_func=None):
    '''
    This function solve PDE, construct derivatives of the solution and some
    other quantities, and compare the results with the analytical values.
                    Input
    PDE             - instance of the PDE class with equation parameters
                      and the calculation settings
    PDESolver       - [None] solver class instance (if is None, then solver is
                      selected according to PDE.solver_txt and PDE.dim values)
    present_res_1s  - [True] if is True, then calculation result will be
                      presented in a one string mode
    return_solver   - [False] if is True, then solver class instance 
                      will be returned
    d               - [None] if is not None, then PDE d-factor will be updated
    apply2pde_func  - [None] if is not None, then this function will be applied
                      to PDE auto solution process, but before check step
                    Output 
    PDESolver       - solver class instance (if return_solver==True)
    '''
    PDE.clean()
    if d is not None:
        PDE.update_d(d)
        
    if PDESolver is None:
        PDESolver = create_solver(PDE) 
        
    PDESolver.solve()
    
    if apply2pde_func is not None:
        apply2pde_func(PDE)
        
    _t = time.time()
    prep_solution(PDE)
    prep_derivative(PDE)
    prep_uu(PDE)
    prep_uf(PDE, PDESolver)
    PDE.t['prep'] = time.time() - _t
    if PDE.verb_gen: 
        PDE._present('Time of additional calcs.  : %-8.4f'%PDE.t['prep']) 
        
    if present_res_1s:
        PDE.present_res_1s()    
        
    if return_solver:
        return PDESolver
        
def prep_solution(PDE):
    '''
    Function construct analytical solution of the PDE (if it is known) 
    and calculate relative error of the calculation result.
    '''
    if PDE.u is not None:
        r = [PDE.GRD.xr, PDE.GRD.yr, PDE.GRD.zr][:PDE.dim]
        PDE.u_real = Vector.func(r, PDE.u_func, None, PDE.verb_crs, 'u_real')                        
        PDE.u_err = PDE.u_real.rel_err(PDE.u_calc)
        PDE.u_real_ranks = PDE.u_real.r
        PDE.r['u_real'] = PDE.u_real.erank

def prep_derivative(PDE):
    '''
    Function construct analytical derivatives of the solution of the PDE 
    (if it is known) and calculate relative errors of the calculation results.
    '''
    if PDE.ux is not None:
        r = [PDE.GRD.xc, PDE.GRD.yr, PDE.GRD.zr][:PDE.dim]
        PDE.ux_real = Vector.func(r, PDE.ux_func, None, PDE.verb_crs, 'ux_real')                        
        PDE.ux_err = PDE.ux_real.rel_err(PDE.ux_calc)
        PDE.r['ux_real'] = PDE.ux_real.erank
    if PDE.uy is not None:
        r = [PDE.GRD.xr, PDE.GRD.yc, PDE.GRD.zr][:PDE.dim]
        PDE.uy_real = Vector.func(r, PDE.uy_func, None, PDE.verb_crs, 'uy_real')                        
        PDE.uy_err = PDE.uy_real.rel_err(PDE.uy_calc)
        PDE.r['uy_real'] = PDE.uy_real.erank
    if PDE.uz is not None:
        r = [PDE.GRD.xr, PDE.GRD.yr, PDE.GRD.zc][:PDE.dim]
        PDE.uz_real = Vector.func(r, PDE.uz_func, None, PDE.verb_crs, 'uz_real')                        
        PDE.uz_err = PDE.uz_real.rel_err(PDE.uz_calc)
        PDE.r['uz_real'] = PDE.uz_real.erank
          
def prep_uu(PDE): 
    '''
    Function construct (u, u) product for calculation result u_calc, and for
    analytical solution u_real (if known), and calculate the relative error.
    '''
    if PDE.u_calc is not None and PDE.u_calc.isnotnone:
        PDE.uu_calc = (PDE.u_calc * PDE.u_calc).sum() * (PDE.h**PDE.dim) 
    if PDE.u_real is not None and PDE.u_real.isnotnone:
        PDE.uu_real = (PDE.u_real * PDE.u_real).sum() * (PDE.h**PDE.dim)
    if PDE.uu_calc is not None and PDE.uu_real is not None:
        PDE.uu_err = norm(PDE.uu_real-PDE.uu_calc)/norm(PDE.uu_real)
        
def prep_uf(PDE, PDESolver): 
    '''
    Function construct (u, f) product for calculation result u_calc, and for
    analytical solution u_real (if known), and calculate the relative error.
    '''
    if PDE.u_calc is not None and PDE.u_calc.isnotnone:
        PDE.uf_calc = (PDE.u_calc * PDESolver.f).sum() * (PDE.h**PDE.dim) 
    if PDE.u_real is not None and PDE.u_real.isnotnone:
        PDE.uf_real = (PDE.u_real * PDESolver.f).sum() * (PDE.h**PDE.dim)
    if PDE.uf_calc is not None and PDE.uf_real is not None:
        PDE.uf_err = norm(PDE.uf_real-PDE.uf_calc)/norm(PDE.uf_real)