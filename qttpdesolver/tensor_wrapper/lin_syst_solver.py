# -*- coding: utf-8 -*- 
import os
import sys
import tempfile

from numpy.linalg import lstsq as np_solve
from tt.amen import amen_solve as tt_solve
from scipy.sparse.linalg import spsolve as sp_solve

from .tensor_base import MODE_NP, MODE_TT, MODE_SP
        
class Capture(object):
    ''' Example for class Capture usage:
            c = Capture()
            c.start_capture()
            os.system('echo 10')
            print('20')
            c.stop_capture()
            print c.out
            print len(c.out)
            # >>> ['10\n', '20\n']
            # >>> 2
    '''
    
    def __init__(self):
        super(Capture, self).__init__()
        self._org = None    # Original stdout stream
        self._dup = None    # Original system stdout descriptor
        self._tmp = None    # Temporary file to write stdout to
        self.clean()
        
    def clean(self):
        self.out = u''
        
    def start_capture(self):
        self._org = sys.stdout
        sys.stdout = sys.__stdout__
        fdout = sys.stdout.fileno()
        self._tmp = tempfile.TemporaryFile()
        self._dup = None
        if fdout >= 0:
            self._dup = os.dup(fdout)
            os.dup2(self._tmp.fileno(), fdout)

    def stop_capture(self):
        sys.stdout.flush()
        if self._dup is not None:
            os.dup2(self._dup, sys.stdout.fileno())
            os.close(self._dup)
        sys.stdout = self._org
        self._tmp.seek(0)
        self.out = self._tmp.readlines()
        self._tmp.close()
   
class CaptureAmen(Capture):
    ''' Example for class CaptureAmen usage:
            import tt
            from tt.amen import amen_solve
            c = CaptureAmen()
            c.start_capture()
            d = 5; n = 2**d
            A = tt.eye(2, d)
            rhs = tt.ones(2, d)
            u0 = tt.rand(rhs.n, r=3)
            tau = 1.E-8
            sol = amen_solve(A, rhs, u0, tau).round(tau)
            c.stop_capture()
            print c.iters
            # >>> 2
    '''
    
    def __init__(self):
        Capture.__init__(self)
      
    @property
    def iters(self):
        count = 0
        for x in self.out:
            if 'amen_solve: swp' in x:
                count+= 1
        return count
        
    @property
    def max_dx(self):
        return float(self.out[-1].split('max_dx= ')[1].split(',')[0])
        
    @property
    def max_res(self):
        return float(self.out[-1].split('max_res= ')[1].split(',')[0])
        
    @property
    def max_rank(self):
        return int(self.out[-1].split('max_rank=')[1])
          
    def present(self, verb):
        if not verb:
            return
        for s in self.out:
            print s,
        
class LinSystSolver(CaptureAmen):
    
    def __init__(self):
        CaptureAmen.__init__(self)
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