# -*- coding: utf-8 -*-
import time

from ..tensor_wrapper import MODE_NP, MODE_TT, MODE_SP, Vector, Matrix

SOLVER_FS = 'fs'
SOLVER_FD = 'fd'
BC_HD = 'hd'
BC_PR = 'pr'
SOLVER_VECTORS = []
SOLVER_VECTORS.extend(['f', 'rhs', 'iqx', 'iqy', 'iqz', 'qx', 'qy', 'qz'])
SOLVER_VECTORS.extend(['wx_calc', 'wy_calc', 'u_calc', 'u_real'])
SOLVER_VECTORS.extend(['ux_calc', 'uy_calc', 'uz_calc'])
SOLVER_VECTORS.extend(['ux_real', 'uy_real', 'uz_real'])
SOLVER_MATRICES =  []
SOLVER_MATRICES.extend(['Bx', 'By', 'Bz', 'iBx', 'iBy', 'iBz'])
SOLVER_MATRICES.extend(['iKx', 'iKy', 'iKz', 'Kx', 'Ky', 'Kz'])
SOLVER_MATRICES.extend(['Wx', 'Wy', 'Wz', 'Rx', 'Ry', 'Rz'])
SOLVER_MATRICES.extend(['Hx', 'Hy', 'Hz', 'A'])
SOLVER_TIMES = ['cgen', 'mgen', 'sgen', 'soln', 'prep']

class Solver(object):
    '''
    Base class for Solver-FS and Solver-FD in 1D/2D/3D cases.
    The input is PDE (an instance of PDE class).
    '''
    
    def __init__(self, PDE):
        self.PDE = PDE
        for name in ['f', 'rhs', 'wx', 'wy', 'u']:
            setattr(self, name, Vector())
        for name in ['u', 'iq', 'q']:
            for dim in ['x', 'y', 'z']:
                setattr(self, name+dim, Vector())
        for name in ['A']:
            setattr(self, name, Matrix())
        for name in ['B', 'iB', 'K', 'iK', 'W', 'R', 'H']:
            for dim in ['x', 'y', 'z']:
                setattr(self, name+dim, Matrix())

    def solve(self):
        self.check()
        self.gen_coefficients()
        self.gen_matrices()
        self.gen_system()
        self.gen_solution()

    def check(self):
        PDE = self.PDE
        if PDE is None or PDE.GRD is None or PDE.LSS is None:
            raise ValueError('PDE, GRD and LSS should be set.')
        if PDE.solver_name and PDE.solver_name not in [SOLVER_FS, SOLVER_FD]:
            raise ValueError('Incorrect name of the solver.')
        if PDE.mode not in [MODE_NP, MODE_TT, MODE_SP]:
            raise ValueError('Incorrect mode.')
        if PDE.bc not in [BC_HD, BC_PR]:
            raise ValueError('Incorrect bc.')
        if PDE.mode == MODE_SP and PDE.solver_name == SOLVER_FS:
            raise ValueError('MODE_SP is not available for Solver-FS.')
        if PDE.bc == BC_PR and PDE.solver_name == SOLVER_FD:
            raise ValueError('BC_PR is not available for Solver-FD.')

    def gen_coefficients(self):
        t = time.time(); PDE = self.PDE
        PDE.GRD.set_params(PDE.d, PDE.h, PDE.dim, PDE.tau, PDE.mode)
        PDE.GRD.construct()
        self._gen_coefficients(PDE, PDE.GRD, PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode, tau=PDE.tau, verb=PDE.verb_crs)
        PDE.t['cgen'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of coeffs.  generation: %-8.4f'%PDE.t['cgen'])
        self.prepare_vars(['f', 'Kx', 'Ky', 'Kz', 'iKx', 'iKy', 'iKz'])

    def gen_matrices(self):
        t = time.time(); PDE = self.PDE
        self._gen_matrices(PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode, tau=PDE.tau, verb=PDE.verb_crs)
        PDE.t['mgen'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of matrices generation: %-8.4f'%PDE.t['mgen'])
        self.prepare_vars(['B', 'iB', 'iq', 'q', 'W', 'R', 'H'], dims=['x', 'y', 'z'])

    def gen_system(self):
        t = time.time(); PDE = self.PDE
        self._gen_system(PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode, tau=PDE.tau, verb=PDE.verb_crs)
        PDE.t['sgen'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of system generation  : %-8.4f'%PDE.t['sgen'])
        self.prepare_vars(['A', 'rhs'])

    def gen_solution(self):
        t = time.time(); PDE = self.PDE
        self._gen_solution(PDE, PDE.LSS, PDE.d, PDE.n, PDE.h, PDE.dim, PDE.mode, eps=PDE.eps_lss, tau=PDE.tau_lss, verb=PDE.verb_lss)
        PDE.t['soln'] = time.time()-t
        if PDE.verb_gen:
            PDE._present('Time of system solving     : %-8.4f'%PDE.t['soln'])
        PDE.u_calc, PDE.u_calc_ranks = self.u, self.u.r
        PDE.ux_calc, PDE.uy_calc, PDE.uz_calc = self.ux, self.uy, self.uz
        self.prepare_vars(['u', 'ux', 'uy', 'uz', 'wx', 'wy'], postf='_calc')

    def prepare_vars(self, names, dims=[''], postf=''):
        for name in names:
            for dim in dims:
                var = getattr(self, name+dim)
                var.name = name+dim+postf
                self.PDE.r[name+dim+postf] = var.erank

def create_solver(PDE):
    if PDE.solver_name==SOLVER_FS and PDE.dim==1:
        from .solver_fs.solver_fs_1d import SolverFS_1d
        return SolverFS_1d(PDE)
    if PDE.solver_name==SOLVER_FS and PDE.dim==2:
        from .solver_fs.solver_fs_2d import SolverFS_2d
        return SolverFS_2d(PDE)
    if PDE.solver_name==SOLVER_FS and PDE.dim==3:
        from .solver_fs.solver_fs_3d import SolverFS_3d
        return SolverFS_3d(PDE)
    if PDE.solver_name==SOLVER_FD and PDE.dim==1:
        from .solver_fd.solver_fd_1d import SolverFD_1d
        return SolverFD_1d(PDE)
    if PDE.solver_name==SOLVER_FD and PDE.dim==2:
        from .solver_fd.solver_fd_2d import SolverFD_2d
        return SolverFD_2d(PDE)
    if PDE.solver_name==SOLVER_FD and PDE.dim==3:
        from .solver_fd.solver_fd_3d import SolverFD_3d
        return SolverFD_3d(PDE)
    raise ValueError('Unknown solver type or incorrect spatial dimension.')
