# -*- coding: utf-8 -*-
ZEROp = 1.E-15
ZEROm =-1.E-15

import unittest
import numpy as np

import tt

from qttpdesolver.utils.general import MODE_NP, MODE_TT, MODE_SP, rel_err
from qttpdesolver.utils.spec_matr import findif
from qttpdesolver.utils.alg_syst_solver import alg_syst_solve

class TestUtilsAlgSystemSolver(unittest.TestCase):
    ''' Tests for functions from utils.alg_syst_solver module (prototype).  '''
    
    def setUp(self):
        self.d = 8
        self.n = 2**self.d
        self.h = 1./self.n
        self.tau = 1.E-8
        self.eps = 1.E-8
        
        self.u = np.arange(self.n)+1.
        self.u_tt = tt.tensor(self.u.reshape([2]*self.d, order='F'), self.tau)
        
        self.A    = findif(self.d, self.n, self.h, self.tau, mode=MODE_NP)
        self.A_sp = findif(self.d, self.n, self.h, self.tau, mode=MODE_SP)
        self.A_tt = findif(self.d, self.n, self.h, self.tau, mode=MODE_TT)
        
        self.rhs = self.A.dot(self.u)
        self.rhs_tt = tt.tensor(self.rhs.reshape([2]*self.d, order='F'), self.tau)
        
    def tearDown(self):
        pass
    
class TestUtilsAlgSystemSolver_alg_syst_solve(TestUtilsAlgSystemSolver):
    ''' 
    Tests for function alg_syst_solve
    from module utils.alg_syst_solver.
    '''
    
    def test_np(self):
        ''' Check for MODE_NP (lstsq solver)  '''
        algss_par = {}
        u1 = alg_syst_solve(self.A, self.rhs, u0=None, eps=None, 
                            algss_par=algss_par, verb=False)
        self.assertTrue(rel_err(u1, self.u) < 3.E-14)
        self.assertTrue(algss_par['iters'] is None)
        self.assertTrue(algss_par['max_dx'] is None)
        self.assertTrue(algss_par['max_res'] is None)
        self.assertTrue(algss_par['max_rank'] is None)
        self.assertTrue(algss_par['solver'] == 'lstsq')
        
    def test_sp(self):
        ''' Check for MODE_NP (direct solver)  '''
        algss_par = {}
        u1 = alg_syst_solve(self.A_sp, self.rhs, u0=None, eps=None, 
                            algss_par=algss_par, verb=False)
        self.assertTrue(rel_err(u1, self.u) < ZEROp)
        self.assertTrue(algss_par['iters'] is None)
        self.assertTrue(algss_par['max_dx'] is None)
        self.assertTrue(algss_par['max_res'] is None)
        self.assertTrue(algss_par['max_rank'] is None)
        self.assertTrue(algss_par['solver'] == 'spsolve')
        
    def test_tt(self):
        ''' Check for MODE_TT (AMEn solver)  '''
        algss_par = {}
        algss_par['nswp'] = 20
        algss_par['kickrank'] = 4
        algss_par['local_prec'] = 'n'
        algss_par['local_iters'] = 2
        algss_par['local_restart'] = 40
        algss_par['max_full_size'] = 50
        algss_par['tau'] = self.eps    
        u2_tt = alg_syst_solve(self.A_tt, self.rhs_tt, None, self.eps, 
                               algss_par)
        self.assertTrue(rel_err(u2_tt, self.u_tt) < self.eps)
#        self.assertTrue(algss_par['iters'] < 10)
#        self.assertTrue(algss_par['max_dx'] < self.eps)
#        self.assertTrue(algss_par['max_res'] < self.eps)
#        self.assertTrue(algss_par['max_rank'] < 10)
        self.assertTrue(algss_par['nswp'] == 20)
        self.assertTrue(algss_par['solver'] == 'amen')
        
if __name__ == '__main__':
    unittest.main()