# -*- coding: utf-8 -*-
ZEROp = 1.E-15
ZEROm =-1.E-15

import unittest
import numpy as np
from numpy.linalg import norm as np_norm
from scipy.sparse import diags as sp_diag

import tt

from qttpdesolver.utils.general import MODE_NP, MODE_TT, MODE_SP
from qttpdesolver.utils.general import is_mode_np, is_mode_tt, is_mode_sp
from qttpdesolver.utils.general import define_mode, define_mode_list
from qttpdesolver.utils.general import rel_err, norm, ttround, vdiag
from qttpdesolver.utils.general import msum, vsum, mprod, vprod, mvec, vinv

class TestUtilsGeneral(unittest.TestCase):
    ''' Tests for functions from utils.general module (prototype).  '''
    
    def setUp(self):
        self.d = 5
        self.n = 2**self.d
        self.tau = 1.E-8
        self.eps = 1.E-8
        self.x = np.arange(self.n)+2.; self.x[2] = -4.
        self.y =-2.*self.x
        self.z = self.x*self.y
        
        self.x_tt = tt.tensor(self.x.reshape([2]*self.d, order='F'), self.tau)
        self.y_tt = tt.tensor(self.y.reshape([2]*self.d, order='F'), self.tau)
        self.z_tt = tt.tensor(self.z.reshape([2]*self.d, order='F'), self.tau)
        
        self.A_tt = tt.Toeplitz(self.x_tt, self.d, kind='U')
        self.B_tt = tt.Toeplitz(self.y_tt, self.d, kind='L')
        self.C_tt = tt.Toeplitz(self.z_tt, self.d, kind='L')
        
        self.A = self.A_tt.full()
        self.B = self.B_tt.full()
        self.C = self.C_tt.full()
        
        self.I = np.eye(self.n)
        self.I_sp = sp_diag([np.ones(self.n)], [0], format='csr')
        
    def tearDown(self):
        pass
    
class TestUtilsGeneral_is_mode_np(TestUtilsGeneral):
    ''' 
    Tests for function is_mode_np.
    from module utils.general.
    '''
        
    def test_correct(self):
        ''' General check.  '''
        self.assertEqual(is_mode_np(1), True)
        self.assertEqual(is_mode_np(2.5), True)
        self.assertEqual(is_mode_np(True), True)
        self.assertEqual(is_mode_np(self.x), True)
        self.assertEqual(is_mode_np(self.A), True)
        self.assertEqual(is_mode_np(self.x_tt), False)
        self.assertEqual(is_mode_np(self.A_tt), False)
        self.assertEqual(is_mode_np(self.I_sp), False)
        self.assertEqual(is_mode_np('abc'), False)
        self.assertEqual(is_mode_np([1, 2, 3]), False)
        self.assertEqual(is_mode_np(self), False)
        
class TestUtilsGeneral_is_mode_tt(TestUtilsGeneral):
    ''' 
    Tests for function is_mode_tt.
    from module utils.general.
    '''
        
    def test_correct(self):
        ''' General check.  '''
        self.assertEqual(is_mode_tt(1), False)
        self.assertEqual(is_mode_tt(2.5), False)
        self.assertEqual(is_mode_tt(True), False)
        self.assertEqual(is_mode_tt(self.x), False)
        self.assertEqual(is_mode_tt(self.A), False)
        self.assertEqual(is_mode_tt(self.x_tt), True)
        self.assertEqual(is_mode_tt(self.A_tt), True)
        self.assertEqual(is_mode_tt(self.I_sp), False)
        self.assertEqual(is_mode_tt('abc'), False)
        self.assertEqual(is_mode_tt([1, 2, 3]), False)
        self.assertEqual(is_mode_tt(self), False)
       
class TestUtilsGeneral_is_mode_sp(TestUtilsGeneral):
    ''' 
    Tests for function is_mode_sp.
    from module utils.general.
    '''
        
    def test_correct(self):
        ''' General check.  '''
        self.assertEqual(is_mode_sp(1), False)
        self.assertEqual(is_mode_sp(2.5), False)
        self.assertEqual(is_mode_sp(True), False)
        self.assertEqual(is_mode_sp(self.x), False)
        self.assertEqual(is_mode_sp(self.A), False)
        self.assertEqual(is_mode_sp(self.x_tt), False)
        self.assertEqual(is_mode_sp(self.A_tt), False)
        self.assertEqual(is_mode_sp(self.I_sp), True)
        self.assertEqual(is_mode_sp('abc'), False)
        self.assertEqual(is_mode_sp([1, 2, 3]), False)
        self.assertEqual(is_mode_sp(self), False)
        
class TestUtilsGeneral_define_mode(TestUtilsGeneral):
    ''' 
    Tests for function define_mode.
    from module utils.general.
    '''
        
    def test_correct(self):
        ''' General check.  '''
        self.assertEqual(define_mode(1), MODE_NP)
        self.assertEqual(define_mode(2.5), MODE_NP)
        self.assertEqual(define_mode(True), MODE_NP)
        self.assertEqual(define_mode(self.x), MODE_NP)
        self.assertEqual(define_mode(self.A), MODE_NP)
        self.assertEqual(define_mode(self.x_tt), MODE_TT)
        self.assertEqual(define_mode(self.A_tt), MODE_TT)
        self.assertEqual(define_mode(self.I_sp), MODE_SP)
        self.assertEqual(define_mode('abc'), None)
        self.assertEqual(define_mode([1, 2, 3]), None)
        self.assertEqual(define_mode(self), None)
        
class TestUtilsGeneral_define_mode_list(TestUtilsGeneral):
    ''' 
    Tests for function define_mode_list.
    from module utils.general.
    '''
        
    def test_correct(self):
        ''' General check.  '''
        self.assertEqual(define_mode_list(['a', 'b', 'c']), None)
        self.assertEqual(define_mode_list([self, self, self]), None)
        self.assertEqual(define_mode_list([1, 2, 3]), MODE_NP)
        self.assertEqual(define_mode_list([self.x, self.y, self.z]), MODE_NP)
        self.assertEqual(define_mode_list([[None, None, None],
                                           [None, None, self.z]]), MODE_NP)
        self.assertEqual(define_mode_list([[None, None, None],
                                           [None, self.z, self.z_tt]]), MODE_NP)
        self.assertEqual(define_mode_list([[None, None, None],
                                           [None, self.z_tt, self.z]]), MODE_TT)
        self.assertEqual(define_mode_list([[None, None, None],
                                           [None, self.I_sp, self.z]]), MODE_SP)
        self.assertEqual(define_mode_list([[self.I_sp, None, None],
                                           [None, self.I_sp, self.z]]), MODE_SP)                                      
        self.assertEqual(define_mode_list([[1, None, None],
                                           [None, self.I_sp, self.z]]), MODE_NP)
        self.assertEqual(define_mode_list([[12.3, None, None],
                                           [None, self.I_sp, self.z]]), MODE_NP)
        self.assertEqual(define_mode_list([[1, 12.3, None],
                                           [None, self.z_tt, self.z]], num_as_none=True), MODE_TT)
        self.assertEqual(define_mode_list([[1, 12.3, None],
                                           [None, None, 42]], num_as_none=True), None)                                           
                                           
class TestUtilsGeneral_norm(TestUtilsGeneral):
    ''' 
    Tests for function norm
    from module utils.general.
    '''
        
    def test_arg_is_none(self):
        ''' If argument is None, the result should be None.  '''
        res = norm(None)
        self.assertEqual(res, None)
        
    def test_ones_vector(self):
        ''' For the vector (100) of ones the result should be 10.  '''
        res = norm(np.ones(100))
        self.assertTrue(abs(res-10.) < ZEROp)
        
    def test_ones_vector_times_2(self):
        ''' For the vector (100) of 2 the result should be 20.  '''
        res = norm(np.ones(100)*2.)
        self.assertTrue(abs(res-20.) < ZEROp)
        
    def test_ones_matrix(self):
        ''' For the matrix (100, 100) of ones the result should be 100.  '''
        res = norm(np.ones((100, 100)))
        self.assertTrue(abs(res-100.) < ZEROp)
        
    def test_ones_matrix_times_2(self):
        ''' For the matrix (100, 100) of 2 the result should be 200.  '''
        res = norm(np.ones((100, 100))*2.)
        self.assertTrue(abs(res-200.) < ZEROp)
        
    def test_np_vs_tt_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = norm(self.x)
        res2 = norm(self.x_tt)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < ZEROp)
    
    def test_np_vs_tt_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = norm(self.A)
        res2 = norm(self.A_tt)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_sp_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = norm(self.I)
        res2 = norm(self.I_sp)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_numbers(self):
        ''' Function should also works correctly for numbers.  '''
        res = norm(12.)
        self.assertTrue(abs(res-12.) < ZEROp)
                                           
class TestUtilsGeneral_rel_err(TestUtilsGeneral):
    ''' 
    Tests for function rel_err
    from module utils.general.
    '''
        
    def test_one_arg_is_none(self):
        ''' If one of arguments is None, the result should be None.  '''
        res = rel_err(self.x, None)
        self.assertEqual(res, None)

    def test_both_arg_is_none(self):
        ''' If both arguments are None, the result should be None.  '''
        res = rel_err(None, None)
        self.assertEqual(res, None)
        
    def test_x_vs_2x(self):
        ''' For the vectors x and 2*x the result should be 1.  '''
        res = rel_err(self.x, 2.*self.x)
        self.assertTrue(abs(res-1.) < ZEROp)
        
    def test_2x_vs_x(self):
        ''' For the vectors 2*x and x the result should be 0.5.  '''
        res = rel_err(2.*self.x, self.x)
        self.assertTrue(abs(res-0.5) < ZEROp)
        
    def test_A_vs_2A(self):
        ''' For the matrices A and 2*A the result should be 1.  '''
        res = rel_err(self.A, 2.*self.A)
        self.assertTrue(abs(res-1.) < ZEROp)
        
    def test_2A_vs_A(self):
        ''' For the matrices 2*A and A the result should be 0.5.  '''
        res = rel_err(2.*self.A, self.A)
        self.assertTrue(abs(res-0.5) < ZEROp)
        
    def test_np_vs_tt_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = rel_err(self.x, self.y)
        res2 = rel_err(self.x_tt, self.y_tt)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < ZEROp)
    
    def test_np_vs_tt_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = rel_err(self.A, self.B)
        res2 = rel_err(self.A_tt, self.B_tt)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_sp_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = rel_err(self.I, self.I*2)
        res2 = rel_err(self.I_sp, self.I_sp*2)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_numbers(self):
        ''' Function should also works correctly for numbers.  '''
        res = rel_err(4., 2.)
        self.assertTrue(abs(res-0.5) < ZEROp)

    def test_exception_for_np_vs_tt_vectors(self):
        ''' In shouldn't work for vectors that differs in type.  '''
        raised = False
        try:
            rel_err(self.x, self.x_tt)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_exception_for_np_vs_tt_matrices(self):
        ''' In shouldn't work for matrices that differs in type.  '''
        raised = False
        try:
            rel_err(self.A_tt, self.A)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_exception_for_np_vs_sp_matrices(self):
        ''' In shouldn't work for matrices that differs in type.  '''
        raised = False
        try:
            rel_err(self.I_sp, self.I)
        except:
            raised = True
        self.assertTrue(raised)
      
class TestUtilsGeneral_ttround(TestUtilsGeneral):
    ''' 
    Tests for function ttround
    from module utils.general.
    '''
    
    def test_arg_is_none(self):
        ''' If argument is None, the result should be None.  '''
        res = ttround(None, self.tau)
        self.assertEqual(res, None)
        
    def test_tau_is_none(self):
        ''' If tau is None, the result should be the same.  '''
        er1 = self.x_tt.erank
        er2 = ttround(self.x_tt, None).erank
        self.assertTrue(er1==er2)
        
    def test_np_is_const(self):
        ''' If argument is in MODE_NP, then out should be equal to in.  '''
        res = ttround(self.x, self.tau)
        eps = rel_err(res, self.x)
        self.assertTrue(eps < ZEROp)
        
    def test_vector(self):
        ''' For higher tolerance erank of vector should be greater.  '''
        er1 = ttround(self.x_tt, tau=1.E-8).erank
        er2 = ttround(self.x_tt, tau=1.E-1).erank
        self.assertTrue(er1>er2)
        
    def test_matrix(self):
        ''' For higher tolerance erank of matrix should be greater.  '''
        er1 = ttround(self.A_tt, tau=1.E-8).erank
        er2 = ttround(self.A_tt, tau=1.E-1).erank
        self.assertTrue(er1>er2)

class TestUtilsGeneral_vdiag(TestUtilsGeneral):
    ''' 
    Tests for function vdiag
    from module utils.general.
    '''
    
    def test_np_vector_is_correct(self):
        ''' For MODE_NP it should be equal to np.diag(x).  '''
        res = vdiag(self.x, self.tau)
        eps = np_norm(res - np.diag(self.x))/np_norm(np.diag(self.x))
        self.assertTrue(eps < ZEROp)
        
    def test_np_matrix_is_correct(self):
        ''' For MODE_NP it should be equal to np.diag(B).  '''
        res = vdiag(self.B, self.tau)
        eps = np_norm(res - np.diag(self.B))/np_norm(np.diag(self.B))
        self.assertTrue(eps < ZEROp)
    
    def test_np_diag2_is_correct(self):
        ''' It we apply function 2 times, we should get the same for vector.  '''
        res = vdiag(self.x)
        res = vdiag(res)
        eps = np_norm(res - self.x)/np_norm(self.x)
        self.assertTrue(eps < ZEROp)

    def test_np_vs_tt_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = vdiag(self.x, self.tau)
        res2 = vdiag(self.x_tt, self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
    
    def test_np_vs_tt_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = vdiag(self.B)
        res2 = vdiag(self.B_tt, self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_sp_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = vdiag(self.x, self.tau)
        res2 = vdiag(self.x, to_sp=True).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
    
    def test_np_vs_sp_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = vdiag(self.I, self.tau)
        res2 = vdiag(self.I_sp)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_incorrect_mode(self):
        ''' If input is not in mode, then result should be None.  '''
        res = vdiag('abc', self.tau)
        self.assertTrue(res is None)
        
class TestUtilsGeneral_msum(TestUtilsGeneral):
    ''' 
    Tests for function msum
    from module utils.general.
    '''
        
    def test_np_vector_is_correct(self):
        ''' For MODE_NP it should be equal to x+y+z.  '''
        res1 = msum([self.x, self.y, self.z])
        res2 = self.x + self.y + self.z
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_matrix_is_correct(self):
        ''' For MODE_NP it should be equal to A+B+C.  '''
        res1 = msum([self.A, self.B, self.C], self.tau)
        res2 = self.A + self.B + self.C
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = msum([self.x, self.y, self.z])
        res2 = msum([self.x_tt, self.y_tt, self.z_tt], self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
    
    def test_np_vs_tt_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = msum([self.A, self.B, self.C], self.tau)
        res2 = msum([self.A_tt, self.B_tt, self.C_tt], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
     
    def test_np_vs_sp_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = msum([self.I]*3, self.tau)
        res2 = msum([self.I_sp]*3).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
class TestUtilsGeneral_vsum(TestUtilsGeneral):
    ''' 
    Tests for function vsum
    from module utils.general.
    '''
        
    def test_np_vector_is_correct(self):
        ''' For MODE_NP it should be equal to np.sum(x).  '''
        res1 = vsum(self.x, self.tau)
        res2 = np.sum(self.x)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = vsum(self.x)
        res2 = vsum(self.x_tt, self.tau)
        eps = abs(res1 - res2)/abs(res1)
        self.assertTrue(eps < 3.E-14)
        
class TestUtilsGeneral_mprod(TestUtilsGeneral):
    ''' 
    Tests for function mprod
    from module utils.general.
    '''
        
    def test_np_matrix_is_correct(self):
        ''' For MODE_NP it should be equal to A.dot(B).dot(C).  '''
        res1 = mprod([self.A, self.B, self.C], self.tau)
        res2 = self.A.dot(self.B).dot(self.C)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
    
    def test_np_vs_tt_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = mprod([self.A, self.B, self.C])
        res2 = mprod([self.A_tt, self.B_tt, self.C_tt], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < 3.E-11)
        
    def test_np_vs_sp_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = mprod([self.I]*3, self.tau)
        res2 = mprod([self.I_sp]*3).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
class TestUtilsGeneral_vprod(TestUtilsGeneral):
    ''' 
    Tests for function vprod
    from module utils.general.
    '''
        
    def test_np_vector_is_correct(self):
        ''' For MODE_NP it should be equal to x*y*z.  '''
        res1 = vprod([self.x, self.y, self.z], self.tau)
        res2 = self.x * self.y * self.z
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = vprod([self.x, self.y, self.z])
        res2 = vprod([self.x_tt, self.y_tt, self.z_tt], self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < 3.E-14)

class TestUtilsGeneral_mvec(TestUtilsGeneral):
    ''' 
    Tests for function mvec
    from module utils.general.
    '''
        
    def test_np_is_correct(self):
        ''' For MODE_NP it should be equal to A.dot(x).  '''
        res1 = mvec(self.A, self.x, self.tau)
        res2 = self.A.dot(self.x)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = mvec(self.A, self.x)
        res2 = mvec(self.A_tt, self.x_tt, self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < 3.E-14)
        
class TestUtilsGeneral_vinv(TestUtilsGeneral):
    ''' 
    Tests for function vinv
    from module utils.general.
    '''
        
    def test_np_is_correct(self):
        ''' For MODE_NP it should be equal to 1/x  '''
        res1 = vinv(self.x)
        res2 = 1./self.x
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
       
    def test_abs_np_is_correct(self):
        ''' For MODE_NP it should be equal to 1/|x|  '''
        res1 = vinv(self.x, abs_val=True)
        res2 = 1./np.abs(self.x)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = vinv(self.x)
        res2 = vinv(self.x_tt, self.eps, self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_abs_np_vs_tt(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = vinv(self.x, abs_val=True)
        res2 = vinv(self.x_tt, self.eps, self.tau, abs_val=True).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
if __name__ == '__main__':
    unittest.main()