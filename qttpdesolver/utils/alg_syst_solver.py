# -*- coding: utf-8 -*- 
import os, sys, tempfile
import numpy as np
from scipy.sparse.linalg import spsolve

from tt.amen import amen_solve

from general import is_mode_np, is_mode_tt, is_mode_sp, ttround
  
def alg_syst_solve(A, rhs, u0, eps, algss_par, verb=False):
    '''
    Solve linear system A u = rhs with initial guess u0, tolerance eps
    and with additional parameters in dictionary algss_par by appropriate
    method (lstsq for MODE_NP, AMEn for MODE_TT and spsolve for MODE_SP).
    The obtained solution is returned, and additional calculation results
    (such as number of iterations) are saved in algss_par.
    '''
    algss_par['iters'] = None
    algss_par['max_dx'] = None
    algss_par['max_res'] = None
    algss_par['max_rank'] = None
    if is_mode_np(A):
        algss_par['solver'] = 'lstsq'
        return np.linalg.lstsq(A, rhs)[0]
    if is_mode_tt(A):
        algss_par['solver'] = 'amen'
        return solve_amen(A, rhs, u0, eps, algss_par, verb)
    if is_mode_sp(A):
        algss_par['solver'] = 'spsolve'
        return spsolve(A, rhs)
    algss_par['solver'] = 'None'
    return None
 
def solve_amen(A, rhs, u0, eps, algss_par, verb=False):
    '''
    Solve linear system A u = rhs with initial guess u0, tolerance eps
    and with additional parameters in dictionary algss_par by AMEn solver.
    The obtained solution is returned, and additional calculation results
    (such as number of iterations) are saved in algss_par.
    '''
    if u0 is None:
        u0 = rhs
    if algss_par.get('nswp') is None:
        algss_par['nswp'] = 20
    if algss_par.get('kickrank') is None:
        algss_par['kickrank'] = 4
    if algss_par.get('local_prec') is None:
        algss_par['local_prec'] = 'n'
    if algss_par.get('local_iters') is None:
        algss_par['local_iters'] = 2
    if algss_par.get('local_restart') is None:
        algss_par['local_restart'] = 40
    if algss_par.get('max_full_size') is None:
        algss_par['max_full_size'] = 50
    if algss_par.get('tau') is None:
        algss_par['tau'] = None
        
    c = CaptureAmen(); c.start()
    u = amen_solve(A, rhs, u0, eps, nswp=algss_par['nswp'],
                   kickrank=algss_par['kickrank'], 
                   local_prec=algss_par['local_prec'],
                   local_iters=algss_par['local_iters'],
                   local_restart=algss_par['local_restart'],
                   max_full_size=algss_par['max_full_size'])   
    c.stop()
    if verb:
        for s in c.get_out():
            print s,
    algss_par['iters'] = c.get_iter_num()
    algss_par['max_dx'] = c.get_max_dx()
    algss_par['max_res'] = c.get_max_res() 
    algss_par['max_rank'] = c.get_max_rank()
    return ttround(u, algss_par['tau'])
    
class Capture(object):
    ''' Example for class Capture usage:
            c = Capture()
            c.start()
            os.system('echo 10')
            print('20')
            c.stop()
            print c.get_out()
            print c.get_out_len()
            # >>> ['10\n', '20\n']
            # >>> 2
    '''
    def __init__(self):
        super(Capture, self).__init__()
        self._org = None    # Original stdout stream
        self._dup = None    # Original system stdout descriptor
        self._file = None   # Temporary file to write stdout to
        self.out = u''
        
    def start(self):
        self._org = sys.stdout
        sys.stdout = sys.__stdout__
        fdout = sys.stdout.fileno()
        self._file = tempfile.TemporaryFile()
        self._dup = None
        if fdout >= 0:
            self._dup = os.dup(fdout)
            os.dup2(self._file.fileno(), fdout)

    def stop(self):
        sys.stdout.flush()
        if self._dup is not None:
            os.dup2(self._dup, sys.stdout.fileno())
            os.close(self._dup)
        sys.stdout = self._org
        self._file.seek(0)
        self.out = self._file.readlines()
        self._file.close()
    
    def get_out(self):
        return self.out
    
    def get_out_len(self):
        return len(self.out)
    
class CaptureAmen(Capture):
    ''' Example for class CaptureAmen usage:
            import tt
            from tt.amen import amen_solve
            c = CaptureAmen()
            c.start()
            d = 5; n = 2**d
            A = tt.eye(2, d)
            rhs = tt.ones(2, d)
            tau = 1.E-8
            u0 = tt.rand(rhs.n, r=3)
            sol = amen_solve(A, rhs, u0, tau).round(tau)
            c.stop()
            print c.get_iter_num()
            # >>> 2
    '''
    def get_iter_num(self):
        count = 0
        for x in self.out:
            if 'amen_solve: swp' in x:
                count+= 1
        return count
        
    def get_max_dx(self):
        return float(self.out[-1].split('max_dx= ')[1].split(',')[0])
        
    def get_max_res(self):
        return float(self.out[-1].split('max_res= ')[1].split(',')[0])
        
    def get_max_rank(self):
        return int(self.out[-1].split('max_rank=')[1])