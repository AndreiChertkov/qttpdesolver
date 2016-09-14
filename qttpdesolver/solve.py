# -*- coding: utf-8 -*-
import time

import tt.multifuncrs2 as multifuncrs2

from utils.general import MODE_TT, ttround, vsum, rel_err, vprod
from utils.grid import quan_on_grid

from solvers.solver_fs import SolverFs
from solvers.solver_fd import SolverFd

def auto_solve(PDE, PDESolver=None, present_res_1s=True, return_solver=False):
    '''
    This function solve PDE, construct derivatives of the solution and some
    other quantities, and compare the result with the analytical values.
                    Input
    PDE             - instance of the PDE class with equation parameters
                      and parameters of the calculation
    PDESolver       - [None] solver class instance (if is None, then solver
                      is selected according to PDE.solver_txt value)
    present_res_1s  - [True] if is True, then calculation result will be
                      presented in a one string mode
    return_solver   - [False] if is True, then solver class instance 
                      will be returned.
                    Output (if return_solver==True)
    PDESolver       - solver class instance 
    '''
    PDE.clean()

    if PDESolver is not None:
        pass
    elif PDE.solver_txt=='fs':
        PDESolver = SolverFs(PDE)
    elif PDE.solver_txt=='fd':
        PDESolver = SolverFd(PDE)
    else:
        raise ValueError('Unknown solver type.')
        
    PDESolver.generate_matrices()
    PDESolver.generate_system()
    PDESolver.solve()
    
    _t = time.time()
    construct_solution(PDE)
    construct_derivative(PDE)
    if PDE.with_en:
        construct_energy(PDE) 
    construct_uf(PDE, PDESolver)
    construct_uu(PDE)
    PDE.t['prepare_res'] = time.time() - _t
    if PDE.verb_gen: 
        PDE._present('Time of additional calcs.  : %-8.4f'%PDE.t['prepare_res']) 
 
    if present_res_1s:
        PDE.present_res_1s()    
    if return_solver:
        return PDESolver
        
def construct_solution(PDE):
    '''
    Function construct analytical solution (if it is known) of the PDE
    and calculate relative error of the calculation result.
    '''
    _t = time.time()
    if PDE.u is None:
        PDE.u_real = None
    else:
        PDE.u_real = quan_on_grid(PDE.u_func, PDE.d, PDE.dim, 
                                  PDE.tau['u_real']['round'],
                                  PDE.tau['u_real']['cross'],
                                  PDE.mode, 'rc', 'u', PDE.verb_cross)                         
    if PDE.u_real is not None and PDE.mode==MODE_TT:
        PDE.u_real_ranks, PDE.u_real_erank = PDE.u_real.r, PDE.u_real.erank
    PDE.t['u_real'] = time.time() - _t
        
    PDE.u_err = rel_err(PDE.u_real, PDE.u_calc)
    
def construct_derivative(PDE):
    '''
    Function construct analytical derivatives of the solution (if it is known) 
    of the PDE and calculate relative error of the calculation result.
    '''
    _t = time.time()
    PDE.ud_real = [None]*PDE.dim
    for i in range(PDE.dim):
        if [PDE.ux, PDE.uy, PDE.uz][i] is None:
            PDE.ud_real = None
            break
        ud_func = [PDE.ux_func, PDE.uy_func, PDE.uz_func][i]
        PDE.ud_real[i] = quan_on_grid(ud_func, PDE.d, PDE.dim, 
                                      PDE.tau['ud_real']['round'], 
                                      PDE.tau['ud_real']['cross'], 
                                      mode=PDE.mode, grid='cc', verb=False)
    PDE.t['ud_real'] = time.time() - _t
    
    if PDE.ud_real is not None and PDE.ud_calc is not None:
        PDE.ud_err = [rel_err(PDE.ud_real[i], PDE.ud_calc[i]) for i in range(PDE.dim)]
        
def construct_energy(PDE): 
    '''
    Function construct analytical energy (if analytical derivatives are known) 
    of the PDE and calculate relative error of the calculation result.
    '''
    _t = time.time()
    if PDE.ud_calc is None and PDE.ud_real is None:
        PDE.t['en'] = time.time()-_t
        return
        
    D = [None]*PDE.dim
    if PDE.dim>=1:
        D[0] = quan_on_grid(PDE.k_func, PDE.d, PDE.dim, 
                            PDE.tau['en_real']['round'], PDE.tau['en_real']['cross'],
                            PDE.mode, 'uxe', 'Dx for e', PDE.verb_cross)
    if PDE.dim>=2:
        D[1] = quan_on_grid(PDE.k_func, PDE.d, PDE.dim, 
                            PDE.tau['en_real']['round'], PDE.tau['en_real']['cross'],
                            PDE.mode, 'uye', 'Dy for e', PDE.verb_cross)
    if PDE.dim>=3:
        D[2] = quan_on_grid(PDE.k_func, PDE.d, PDE.dim, 
                            PDE.tau['en_real']['round'], PDE.tau['en_real']['cross'],
                            PDE.mode, 'uze', 'Dz for e', PDE.verb_cross)
                            
    def calc_en(ud, K, tau_cross, tau_round, mode=MODE_TT):
        if mode != MODE_TT:
            e = ud*K*ud
        else:
            e = multifuncrs2([ud, K, ud], lambda x: x[:, 0]*x[:, 1]*x[:, 2], 
                             tau_cross, verb=False)
        e = ttround(e, tau_round)
        e = vsum(e, tau_round)
        return e 

    if PDE.ud_real is not None:
        PDE.en_real = 0.
        for i in range(PDE.dim):
            PDE.en_real+= calc_en(PDE.ud_real[i], D[i], 
                                  PDE.tau['en_real']['cross'], 
                                  PDE.tau['en_real']['round'], PDE.mode)
        PDE.en_real*= (PDE.h**PDE.dim)

    if PDE.ud_calc is not None:
        PDE.en_calc = 0.
        for i in range(PDE.dim):
            PDE.en_calc+= calc_en(PDE.ud_calc[i], D[i],
                                  PDE.tau['en_calc']['cross'], 
                                  PDE.tau['en_calc']['round'], PDE.mode)
        PDE.en_calc*= (PDE.h**PDE.dim)

    PDE.en_err = rel_err(PDE.en_real, PDE.en_calc)
    PDE.t['en'] = time.time()-_t
    
def construct_uf(PDE, PDESolver): 
    '''
    Function construct (u, f) product for calculation result u_calc, and for
    analytical solution u_real (if known), and calculate the relative error.
    '''
    tau_round = PDE.tau['en_calc']['round']
    
    _t = time.time()
    if PDE.u_calc is not None:
        uf = vprod([PDE.u_calc, PDESolver.F], tau_round)
        PDE.uf_calc = vsum(uf, tau_round) * (PDE.h**PDE.dim)
    PDE.t['uf_calc'] = time.time()-_t

    _t = time.time()
    if PDE.u_real is not None:
        uf = vprod([PDE.u_real, PDESolver.F], tau_round)
        PDE.uf_real = vsum(uf, tau_round) * (PDE.h**PDE.dim)
    PDE.t['uf_real'] = time.time()-_t

    PDE.uf_err = rel_err(PDE.uf_real, PDE.uf_calc)
    
def construct_uu(PDE): 
    '''
    Function construct (u, u) product for calculation result u_calc, and for
    analytical solution u_real (if known), and calculate the relative error.
    '''
    tau_round = PDE.tau['en_calc']['round']
    
    _t = time.time()
    if PDE.u_calc is not None:
        uu = vprod([PDE.u_calc, PDE.u_calc], tau_round)
        PDE.uu_calc = vsum(uu, tau_round) * (PDE.h**PDE.dim)
    PDE.t['uu_calc'] = time.time()-_t

    _t = time.time()
    if PDE.u_real is not None:
        uu = vprod([PDE.u_real, PDE.u_real], tau_round)
        PDE.uu_real = vsum(uu, tau_round) * (PDE.h**PDE.dim)
    PDE.t['uu_real'] = time.time()-_t

    PDE.uu_err = rel_err(PDE.uu_real, PDE.uu_calc)