# -*- coding: utf-8 -*-
import os
import sys
import tempfile
from numpy.linalg import lstsq as np_solve
from tt.amen import amen_solve as tt_solve
from scipy.sparse.linalg import spsolve as sp_solve

from . import MODE_NP, MODE_TT, MODE_SP
from ..utils.capture_output import CaptureAmen

class LinSystSolver(CaptureAmen):

    def __init__(self):
        super(LinSystSolver, self).__init__()
        self.solver = None
        self.set_params()

    def set_params(self, nswp=20, kickrank=4, local_prec='n', local_iters=2,
                   local_restart=40, trunc_norm=1, max_full_size=50):
        self.nswp = nswp
        self.kickrank = kickrank
        self.local_prec = local_prec
        self.local_iters = local_iters
        self.local_restart = local_restart
        self.trunc_norm = trunc_norm
        self.max_full_size = max_full_size

    def solve(self, A, rhs, eps, tau=None, u0=None, verb=False):
        '''
        Solve linear system A u = rhs with initial guess u0 and tolerance eps
        by appropriate method:
            MODE_NP: np.linalg.lstsq
            MODE_TT: tt.amen.amen_solve
            MODE_SP: scipy.sparse.linalg.spsolve
        If u0 is None, then it is set to rhs value.
        If tau is not given, then tau is set to real accuracy of solver output.
        The obtained solution is rounded to accuracy tau.
        '''
        u = rhs.copy(copy_x=False)
        u.name = 'Lin. syst. solution'
        if A.mode == MODE_NP:
            self.solver = 'lstsq'
            u.x = np_solve(A.x, rhs.x)[0]
        elif A.mode == MODE_TT:
            if u0 is None:
                u0 = rhs.copy()
            self.solver = 'amen'
            self.start_capture()
            u.x = tt_solve(A.x, rhs.x, u0.x, eps,
                           nswp=self.nswp,
                           kickrank=self.kickrank,
                           local_prec=self.local_prec,
                           local_iters=self.local_iters,
                           local_restart=self.local_restart,
                           trunc_norm=self.trunc_norm,
                           max_full_size=self.max_full_size)
            self.stop_capture()
            if tau is None:
                tau = self.max_res
            u = u.round(tau)
            self.present(verb)
        elif A.mode == MODE_SP:
            self.solver = 'spsolve'
            u.x = sp_solve(A.x, rhs.x)
        else:
            self.solver = None
            raise ValueError('Incorect mode of the input.')
        return u

    def copy(self):
        LSS = LinSystSolver()
        LSS.out = self.out
        LSS.solver = self.solver
        LSS.set_params(self.nswp, self.kickrank, self.local_prec, self.local_iters, self.local_restart, self.trunc_norm, self.max_full_size)
        return LSS
