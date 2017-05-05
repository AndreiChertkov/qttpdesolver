# -*- coding: utf-8 -*-
SOLVER_FS    = 'fs'
SOLVER_FD    = 'fd'
SOLVER_FSX   = 'fsx'
SOLVER_FDX   = 'fdx'
SOLVER_FS_NH = 'fs-nh'
BC_HD        = 'hd'
BC_PR        = 'pr'

import time

from ..tensor_wrapper import MODE_NP, MODE_TT, MODE_SP, Vector, Matrix

class Solver(object):
    '''
    Base class for Solver-FS and Solver-FD in 1D/2D/3D cases.
    The input is PDE (an instance of PDE class).
    '''  
    
    def __init__(self, PDE):
        self.set_pde(PDE)
        self.clean()

    def set_pde(self, PDE):
        self.PDE = PDE
        self.name = self.PDE.solver_txt
        if self.name is not None and self.name not in [SOLVER_FS, SOLVER_FS_NH, SOLVER_FD, SOLVER_FSX, SOLVER_FDX]:
            raise ValueError('Incorrect name of the solver.')
        self.mode = self.PDE.mode
        if self.mode == MODE_SP and self.name in [SOLVER_FS, SOLVER_FS_NH, SOLVER_FSX]:
           raise ValueError('MODE_SP is not available for Solver-FS.')
        self.bc = self.PDE.bc
        if self.bc == BC_PR and self.name in [SOLVER_FD, SOLVER_FDX]:
           raise ValueError('BC_PR is not available for Solver-FD.')
           
    def clean(self):
        for name in ['f', 'rhs', 'wx', 'wy', 'u']:
            setattr(self, '%s'%(name), Vector())
        for name in ['u', 'iq', 'q']:
            for dim in ['x', 'y', 'z']:
                setattr(self, '%s%s'%(name, dim), Vector())
        for name in ['A']:
            setattr(self, '%s'%(name), Matrix())
        for name in ['B', 'iB', 'K', 'iK', 'W', 'R', 'H']:
            for dim in ['x', 'y', 'z']:
                setattr(self, '%s%s'%(name, dim), Matrix())
        
    def solve(self):
        if self.PDE is None or self.PDE.GRD is None or self.PDE.LSS is None:
            raise ValueError('PDE, GRD and LSS should be set.')
        self.gen_coefficients()
        self.gen_matrices()
        self.gen_system()
        self.gen_solution()
    
    def gen_coefficients(self):
        t = time.time(); PDE = self.PDE
        PDE.GRD.set_params(PDE.d, PDE.h, PDE.dim, PDE.tau, PDE.mode)
        PDE.GRD.construct()
        self._gen_coefficients(PDE, PDE.GRD, PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode, 
                               tau=PDE.tau, verb=PDE.verb_crs)
        PDE.t['cgen'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of coeffs.  generation: %-8.4f'%PDE.t['cgen'])
            
        for name in ['f', 'Kx', 'Ky', 'Kz', 'iKx', 'iKy', 'iKz']:
            var = eval('self.%s'%name)
            var.name = name
            PDE.r[name] = var.erank
            
    def gen_matrices(self):
        t = time.time(); PDE = self.PDE
        self._gen_matrices(PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode,  
                           tau=PDE.tau, verb=PDE.verb_crs)
        PDE.t['mgen'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of matrices generation: %-8.4f'%PDE.t['mgen'])
            
        for name in ['B', 'iB', 'iq', 'q', 'W', 'R', 'H']:
            for dim in ['x', 'y', 'z']:
                var = eval('self.%s'%(name+dim))
                var.name = (name+dim)
                PDE.r[(name+dim)] = var.erank
            
    def gen_system(self):
        t = time.time(); PDE = self.PDE
        self._gen_system(PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode, 
                         tau=PDE.tau, verb=PDE.verb_crs)
        PDE.t['sgen'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of system generation  : %-8.4f'%PDE.t['sgen'])

        for name in ['A', 'rhs']:
            var = eval('self.%s'%name)
            var.name = name
            PDE.r[name] = var.erank
                      
    def gen_solution(self):
        t = time.time(); PDE = self.PDE
        self._gen_solution(PDE, PDE.LSS, PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode, 
                           eps=PDE.eps_lss, tau=PDE.tau_lss, verb=PDE.verb_lss)
        PDE.t['soln'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of system solving     : %-8.4f'%PDE.t['soln'])

        PDE.u_calc, PDE.u_calc_ranks = self.u, self.u.r
        PDE.ux_calc, PDE.uy_calc, PDE.uz_calc = self.ux, self.uy, self.uz
        for name in ['u', 'ux', 'uy', 'uz', 'wx', 'wy']:
            var = eval('self.%s'%name)
            var.name = name+'_calc'
            PDE.r[name+'_calc'] = var.erank
 
def create_solver(PDE):
    if PDE.solver_txt==SOLVER_FS and PDE.dim==1:
        from .solver_fs.solver_fs_1d import SolverFS_1d
        return SolverFS_1d(PDE)
    if PDE.solver_txt==SOLVER_FS and PDE.dim==2:
        from .solver_fs.solver_fs_2d import SolverFS_2d
        return SolverFS_2d(PDE)
    if PDE.solver_txt==SOLVER_FS and PDE.dim==3:
        from .solver_fs.solver_fs_3d import SolverFS_3d
        return SolverFS_3d(PDE)
    if PDE.solver_txt==SOLVER_FS_NH and PDE.dim==1:
        from .solver_fs_nh.solver_fs_nh_1d import SolverFS_NH_1d
        return SolverFS_NH_1d(PDE)
    if PDE.solver_txt==SOLVER_FD and PDE.dim==1:
        from .solver_fd.solver_fd_1d import SolverFD_1d
        return SolverFD_1d(PDE)
    if PDE.solver_txt==SOLVER_FD and PDE.dim==2:
        from .solver_fd.solver_fd_2d import SolverFD_2d
        return SolverFD_2d(PDE)
    if PDE.solver_txt==SOLVER_FD and PDE.dim==3:
        from .solver_fd.solver_fd_3d import SolverFD_3d
        return SolverFD_3d(PDE)
    if PDE.solver_txt==SOLVER_FSX and PDE.dim==2:
        from .solver_fsx.solver_fsx_2d import SolverFSX_2d
        return SolverFSX_2d(PDE)
    if PDE.solver_txt==SOLVER_FDX and PDE.dim==2:
        from .solver_fdx.solver_fdx_2d import SolverFDX_2d
        return SolverFDX_2d(PDE)
    raise ValueError('Unknown solver type or incorrect spatial dimension.')