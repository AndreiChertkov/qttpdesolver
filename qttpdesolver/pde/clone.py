# -*- coding: utf-8 -*-
from copy import deepcopy

def clone(PDE0, with_sol=False, with_sol0=False, with_der=False):     
    PDE = Pde(PDE0.txt, PDE0.dim, PDE0.bc, PDE0.mode, LSS=PDE0.LSS.copy())
    PDE.set_print(PDE0.print_to_std, PDE0.print_to_file, PDE0.out_file)
    PDE.L = PDE0.L
    PDE.model_name = PDE0.model_name
    PDE.set_params(deepcopy(PDE0.params), deepcopy(PDE0.params_names), deepcopy(PDE0.params_forms))

    PDE.k_txt            = PDE0.k_txt
    PDE.f_txt            = PDE0.f_txt
    PDE.u_txt            = PDE0.u_txt
    PDE.ux_txt           = PDE0.ux_txt
    PDE.uy_txt           = PDE0.uy_txt
    PDE.uz_txt           = PDE0.uz_txt
    if with_grd: PDE.GRD = PDE0.GRD.copy()

    PDE.set_solver_txt(PDE0.solver_txt)
    PDE.set_verb(PDE0.verb_gen, PDE0.verb_crs, PDE0.verb_lss)
    if with_sol0 and PDE0.sol0 is not None: PDE.set_sol0(PDE0.sol0.copy())
    PDE.set_tau(PDE0.tau, PDE0.eps_lss, PDE0.tau_lss, PDE0.tau_real)
    PDE.update_d(PDE0.d)
    PDE.t = deepcopy(PDE0.t)
    PDE.r = deepcopy(PDE0.r)    
    if with_u_real and PDE0.u_real is not None: PDE.u_real = PDE0.u_real.copy()
    PDE.u_real_ranks = deepcopy(PDE0.u_real_ranks)
    if with_u_calc and PDE0.u_calc is not None: PDE.u_calc = PDE0.u_calc.copy()
    PDE.u_calc_ranks = deepcopy(PDE0.u_calc_ranks)
    PDE.u_err = PDE0.u_err      
    if with_ux_real and PDE0.ux_real is not None: PDE.ux_real = PDE0.ux_real.copy()
    if with_uy_real and PDE0.uy_real is not None: PDE.uy_real = PDE0.uy_real.copy()
    if with_uz_real and PDE0.uz_real is not None: PDE.uz_real = PDE0.uz_real.copy()
    if with_ux_calc and PDE0.ux_calc is not None: PDE.ux_calc = PDE0.ux_calc.copy()
    if with_uy_calc and PDE0.uy_calc is not None: PDE.uy_calc = PDE0.uy_calc.copy()
    if with_uz_calc and PDE0.uz_calc is not None: PDE.uz_calc = PDE0.uz_calc.copy()
    PDE.ux_err  = PDE0.ux_err
    PDE.uy_err  = PDE0.uy_err
    PDE.uz_err  = PDE0.uz_err
    PDE.uu_real = PDE0.uu_real
    PDE.uu_calc = PDE0.uu_calc
    PDE.uu_err  = PDE0.uu_err
    PDE.uf_real = PDE0.uf_real
    PDE.uf_calc = PDE0.uf_calc
    PDE.uf_err  = PDE0.uf_err
    if with_funcs:
        PDE.k  = PDE0.k
        PDE.f  = PDE0.f
        PDE.u  = PDE0.u
        PDE.ux = PDE0.ux
        PDE.uy = PDE0.uy
        PDE.uz = PDE0.uz
    return PDE