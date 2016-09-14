# -*- coding: utf-8 -*-
ZEROp = 1.E-15
ZEROm =-1.E-15

import unittest
import numpy as np

import tt

from qttpdesolver.utils.general import MODE_NP, MODE_TT, MODE_SP, rel_err
from qttpdesolver.utils.spec_matr import toeplitz, eye, volterra, findif
from qttpdesolver.utils.spec_matr import shift, interpol_1d, vzeros_except_one

class TestUtilsSpecMatr(unittest.TestCase):
    ''' Tests for functions from utils.spec_matr module (prototype).  '''
    
    def setUp(self):
        self.d = 4
        self.n = 2**self.d
        self.h = 1./self.n
        self.tau = 1.E-8
        self.eps = 1.E-8
        self.x = np.arange(self.n)+2.; self.x[2] = -4.
        self.y =-2.*self.x
        
        self.x_tt = tt.tensor(self.x.reshape([2]*self.d, order='F'), self.tau)
        self.y_tt = tt.tensor(self.y.reshape([2]*self.d, order='F'), self.tau)
        
    def tearDown(self):
        pass
    
class TestUtilsSpecMatr_toeplitz(TestUtilsSpecMatr):
    ''' 
    Tests for function toeplitz
    from module utils.spec_matr.
    '''
        
    def test_correct(self):
        ''' The result should be correct.  '''
        T = toeplitz(self.x, self.y)
        self.assertTrue(rel_err(T[0, 0], self.x[0]) < ZEROp)
        self.assertTrue(rel_err(T[2, 0], self.x[2]) < ZEROp)
        self.assertTrue(rel_err(T[-1, 0], self.x[-1]) < ZEROp)
        self.assertTrue(rel_err(T[0, 2], self.y[2]) < ZEROp)
        self.assertTrue(rel_err(T[0, -1], self.y[-1]) < ZEROp)
        self.assertTrue(rel_err(T[1, -1], self.y[-2]) < ZEROp)
        self.assertTrue(rel_err(T[2, 1], self.x[1]) < ZEROp)
        
    def test_row(self):
        ''' The first element in the input r shouldn't affect the result.  '''
        T1 = toeplitz(self.x, self.y)
        self.y[0] = None
        T2 = toeplitz(self.x, self.y)
        self.assertTrue(rel_err(T1, T2) < ZEROp)
        
class TestUtilsSpecMatr_eye(TestUtilsSpecMatr):
    ''' 
    Tests for function eye
    from module utils.spec_matr.
    '''
        
    def test_np_sum(self):
        ''' Sum of elements should be n.  '''
        B = eye(self.d, self.n)
        self.assertTrue(abs(np.sum(B)-(self.n)) < ZEROp)

    def test_np_correct(self):
        ''' The matrix form is checked.  '''
        B = eye(self.d, self.n, mode=MODE_NP)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if j==i:
                    self.assertTrue(abs(B[i, j] - 1.) < ZEROp)
                else:
                    self.assertTrue(abs(B[i, j]) < ZEROp)

    def test_np_vs_sp(self):
        ''' The results for MODE_NP and MODE_SP should be equal.  '''
        B1 = eye(self.d, self.n)
        B2 = eye(self.d, self.n, mode=MODE_SP).toarray()
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
    def test_np_vs_tt(self):
        ''' The results for MODE_NP and MODE_TT should be equal.  '''
        B1 = eye(self.d, self.n, MODE_NP)
        B2 = eye(self.d, self.n, MODE_TT).full()
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
class TestUtilsSpecMatr_volterra(TestUtilsSpecMatr):
    ''' 
    Tests for function volterra
    from module utils.spec_matr.
    '''
        
    def test_np_sum(self):
        ''' Sum of elements should be h*(n*n+n)/2 = (n+1)/2.  '''
        B = volterra(self.d, self.n, self.h)
        self.assertEqual(np.sum(B), (self.n + 1.)/2)

    def test_np_correct(self):
        ''' The matrix form is checked.  '''
        B = volterra(self.d, self.n, self.h, self.tau)
        for i in range(B.shape[0]):
            for j in range(i+1):
                self.assertTrue(abs(B[i, j] - self.h) < ZEROp)
            for j in range(i+1, B.shape[1]):
                self.assertTrue(abs(B[i, j] - 0.) < ZEROp)

    def test_np_vs_sp(self):
        ''' The results for MODE_NP and MODE_SP should be equal.  '''
        B1 = volterra(self.d, self.n, self.h, self.tau, MODE_NP)
        B2 = volterra(self.d, self.n, self.h, MODE_SP)
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
    def test_np_vs_tt(self):
        ''' The results for MODE_NP and MODE_TT should be equal.  '''
        B1 = volterra(self.d, self.n, self.h, self.tau, MODE_NP)
        B2 = volterra(self.d, self.n, self.h, self.tau, MODE_TT).full()
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
class TestUtilsSpecMatr_findif(TestUtilsSpecMatr):
    ''' 
    Tests for function findif
    from module utils.spec_matr.
    '''
        
    def test_np_sum(self):
        ''' Sum of elements should be 1/h  = n.  '''
        B = findif(self.d, self.n, self.h)
        self.assertTrue(abs(np.sum(B)-self.n) < ZEROp)

    def test_np_correct(self):
        ''' The matrix form is checked.  '''
        B = findif(self.d, self.n, self.h, self.tau)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if j==i-1:
                    self.assertTrue(abs(B[i, j] + self.n) < ZEROp)
                elif j==i:
                    self.assertTrue(abs(B[i, j] - self.n) < ZEROp)
                else:
                    self.assertTrue(abs(B[i, j]) < ZEROp)

    def test_np_vs_sp(self):
        ''' The results for MODE_NP and MODE_SP should be equal.  '''
        B1 = findif(self.d, self.n, self.h, self.tau, MODE_NP)
        B2 = findif(self.d, self.n, self.h, mode=MODE_SP).toarray()
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
    def test_np_vs_tt(self):
        ''' The results for MODE_NP and MODE_TT should be equal.  '''
        B1 = findif(self.d, self.n, self.h, mode=MODE_NP)
        B2 = findif(self.d, self.n, self.h, self.tau, MODE_TT).full()
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
class TestUtilsSpecMatr_shift(TestUtilsSpecMatr):
    ''' 
    Tests for function shift
    from module utils.spec_matr.
    '''
        
    def test_np_sum(self):
        ''' Sum of elements should be n-1.  '''
        B = shift(self.d, self.n)
        self.assertTrue(abs(np.sum(B)-(self.n-1)) < ZEROp)

    def test_np_correct(self):
        ''' The matrix form is checked.  '''
        B = shift(self.d, self.n, mode=MODE_NP)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if j==i-1:
                    self.assertTrue(abs(B[i, j] - 1.) < ZEROp)
                else:
                    self.assertTrue(abs(B[i, j]) < ZEROp)

    def test_np_vs_sp(self):
        ''' The results for MODE_NP and MODE_SP should be equal.  '''
        B1 = shift(self.d, self.n)
        B2 = shift(self.d, self.n, mode=MODE_SP).toarray()
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
    def test_np_vs_tt(self):
        ''' The results for MODE_NP and MODE_TT should be equal.  '''
        B1 = shift(self.d, self.n, self.tau, MODE_NP)
        B2 = shift(self.d, self.n, self.tau, MODE_TT).full()
        self.assertTrue(rel_err(B1, B2) < self.tau)
        
class TestUtilsSpecMatr_interpol_1d(TestUtilsSpecMatr):
    ''' 
    Tests for function interpol_1d
    from module utils.spec_matr.
    '''
    
    def test_np_sum(self):
        ''' Sum of elements should be 2n-1/2.  '''
        B = interpol_1d(self.d, self.n)
        self.assertTrue(abs(np.sum(B)-(2*self.n-0.5)) < ZEROp)

    def test_np_correct(self):
        ''' The matrix form is checked.  '''
        B = interpol_1d(self.d, self.n, mode=MODE_NP)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                if j%2==1:
                    self.assertTrue(abs(B[i, j]) < ZEROp)
                elif i==j:
                    self.assertTrue(abs(B[i, j]-0.5) < ZEROp)
                    self.assertTrue(abs(B[i+1, j]-1.) < ZEROp)
                   
    def test_sp_not_work(self):
        ''' In shouldn't work for MODE_SP.  '''
        raised = False
        try:
            interpol_1d(self.d, self.n, mode=MODE_SP)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_vs_tt(self):
        ''' The results for MODE_NP and MODE_TT should be equal.  '''
        B1 = interpol_1d(self.d, self.n, self.tau, MODE_NP)
        B2 = interpol_1d(self.d, self.n, self.tau, MODE_TT).full()
        self.assertTrue(rel_err(B1, B2) < ZEROp)
        
class TestUtilsSpecMatr_vzeros_except_one(TestUtilsSpecMatr):
    ''' 
    Tests for function vzeros_except_one
    from module utils.spec_matr.
    '''
    
    def test_np(self):
        ''' The results should be correct for MODE_NP.  '''
        d = 3
        n = 2**d
        value = 2.
        for ind in [0, 2, 4, n-1]:
            res = vzeros_except_one(d, ind, mode=MODE_NP, value=value)
            self.assertEqual(len(res), n)
            for i in range(n):
                if i != ind:
                    self.assertTrue(abs(res[i]) < ZEROp)
                else:
                    self.assertTrue(abs(res[i]-value) < ZEROp)

    def test_np_vs_tt(self):
        ''' The results for MODE_NP and MODE_TT should be equal.  '''
        d = 3
        n = 2**d
        value = 2.
        for ind in [0, 2, 4, n-1]:
            res1 = vzeros_except_one(d, ind, mode=MODE_NP, value=value)
            res2 = vzeros_except_one(d, ind, mode=MODE_TT, value=value).full().flatten('F')
            self.assertTrue(rel_err(res1, res2) < ZEROp)
     
if __name__ == '__main__':
    unittest.main()