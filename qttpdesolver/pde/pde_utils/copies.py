# -*- coding: utf-8 -*-
from copy import deepcopy

def copy(PDE, with_u_real=False, with_u_calc=False):
    PDE_tmp = deepcopy(PDE)
    if not with_u_real:
        PDE_tmp.u_real = None
    if not with_u_calc:
        PDE_tmp.u_calc = None
    PDE_tmp.sol0    = None
    PDE_tmp.ud_calc = None
    PDE_tmp.ud_real = None
    PDE_tmp.f = None
    PDE_tmp.k = None
    PDE_tmp.u = None
    PDE_tmp.ux= None
    PDE_tmp.uy= None
    PDE_tmp.uz= None    
    return PDE_tmp 