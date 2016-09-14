# -*- coding: utf-8 -*-
ZEROp = 1.E-15
ZEROm =-1.E-15

import unittest
import numpy as np
from numpy.linalg import norm as np_norm
from scipy.sparse import diags as sp_diag

import tt

from qttpdesolver.utils.block_and_space import mblock, vblock
from qttpdesolver.utils.block_and_space import kronm, space_kron
from qttpdesolver.utils.block_and_space import sum_out, kron_out

class TestUtilsBlockAndSpace(unittest.TestCase):
    ''' Tests for functions from utils.block_and_space module (prototype).  '''
    
    def setUp(self):
        self.d = 3
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
            
class TestUtilsBlockAndSpace_mblock(TestUtilsBlockAndSpace):
    ''' 
    Tests for function mblock
    from module utils.block_and_space.
    '''

    def test_exception_for_incorrect_input(self):
        raised = False
        try:
            mblock(self.A, self.B)
        except:
            raised = True
        self.assertTrue(raised)
        
        raised = False
        try:
            mblock([self.A, [self.B]])
        except:
            raised = True
        self.assertTrue(raised)
        raised = False
        try:
            mblock([[self.A, self.A], [self.B]])
        except:
            raised = True
        self.assertTrue(raised)
        raised = False
        try:
            mblock([np.eye(3), np.eye(4)])
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_is_correct(self):
        ''' The result should be correct.  '''
        n, m, shn, shm = 3, 4, 4, 4
        for i in range(n):
            for j in range(m):
                lst = [[np.random.randn(shn, shm)]*m for _ in range(n)]
                lst[i][j] = None
                M = mblock(lst)
                for i1 in range(n):
                    for j1 in range(m):
                        B = M[shn*i1:shn*(i1+1), shm*j1:shm*(j1+1)]
                        if i1 != i or j1 != j: 
                            eps = np_norm(B - lst[i1][j1])/np_norm(B)
                            self.assertTrue(eps < ZEROp)
                        else:
                            eps = np.max(np.abs(B))
                            self.assertTrue(eps < ZEROp)
        n, m, shn, shm = 2, 3, 3, 3
        for i in range(n-1):
            for j in range(m):
                lst = [[np.random.randn(shn, shm)]*m for _ in range(n)]
                lst[i][j] = None
                lst[i+1][j] = 1.
                M = mblock(lst)
                for i1 in range(n):
                    for j1 in range(m):
                        B = M[shn*i1:shn*(i1+1), shm*j1:shm*(j1+1)]
                        if (i1 != i and i1 != i+1) or j1 != j: 
                            eps = np_norm(B - lst[i1][j1])/np_norm(B)
                            self.assertTrue(eps < ZEROp)
                        else:
                            eps = np.max(np.abs(B))
                            self.assertTrue(eps < ZEROp)
        n, m, shn, shm = 2, 3, 3, 3
        for i in range(n):
            for j in range(1, m):
                lst = [[np.random.randn(shn, shm)]*m for _ in range(n)]
                lst[i][j] = None
                lst[i][j-1] = 2
                M = mblock(lst)
                for i1 in range(n):
                    for j1 in range(m):
                        B = M[shn*i1:shn*(i1+1), shm*j1:shm*(j1+1)]
                        if i1 != i or (j1 != j-1 and j1 != j): 
                            eps = np_norm(B - lst[i1][j1])/np_norm(B)
                            self.assertTrue(eps < ZEROp)
                        else:
                            eps = np.max(np.abs(B))
                            self.assertTrue(eps < ZEROp)
        for i in range(n):
            lst = [[np.random.randn(shn, shm)]*m for _ in range(n)]
            lst[i] =[None for _ in range(m)]
            M = mblock(lst)
            for i1 in range(n):
                for j1 in range(m):
                    B = M[shn*i1:shn*(i1+1), shm*j1:shm*(j1+1)]
                    if i1 != i: 
                        eps = np_norm(B - lst[i1][j1])/np_norm(B)
                        self.assertTrue(eps < ZEROp)
                    else:
                        eps = np.max(np.abs(B))
                        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt_1(self):
        ''' In the both modes the results should be the same for [[A, B], [A, C]]  '''
        res1 = mblock([[self.A, self.B], [self.A, self.C]])
        res2 = mblock([[self.A_tt, self.B_tt], [self.A_tt, self.C_tt]], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)  
        
    def test_np_vs_tt_2(self):
        ''' In the both modes the results should be the same for [[0, B], [A, C]]  '''
        res1 = mblock([[0, self.B], [self.A, self.C]])
        res2 = mblock([[None, self.B_tt], [self.A_tt, self.C_tt]], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)  
        
    def test_np_vs_tt_3(self):
        ''' In the both modes the results should be the same for [[A, 0], [B, C]]  '''
        res1 = mblock([[self.A, None], [self.B, self.C]])
        res2 = mblock([[self.A_tt, 2], [self.B_tt, self.C_tt]], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)  
        
    def test_np_vs_tt_4(self):
        ''' In the both modes the results should be the same for [[A, B], [0, C]]  '''
        res1 = mblock([[self.A, self.B], [0, self.C]])
        res2 = mblock([[self.A_tt, self.B_tt], [None, self.C_tt]], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)  
        
    def test_np_vs_tt_5(self):
        ''' In the both modes the results should be the same for [[A, B], [C, 0]]  '''
        res1 = mblock([[self.A, self.B], [self.C, 0]])
        res2 = mblock([[self.A_tt, self.B_tt], [self.C_tt, 0]], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau) 
 
    def test_np_vs_tt_6(self):
        ''' In the both modes the results should be the same for [[A, B], [A, C]]  '''
        res1 = mblock([[self.A, self.B, None], [self.A, None, self.C]])
        res2 = mblock([[self.A_tt, self.B_tt, 2], [self.A_tt, 2., self.C_tt]], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)

class TestUtilsBlockAndSpace_vblock(TestUtilsBlockAndSpace):
    ''' 
    Tests for function vblock
    from module utils.block_and_space.
    '''

    def test_exception_for_incorrect_input(self):
        raised = False
        try:
            vblock(self.x, self.y)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_is_correct(self):
        ''' The result should be correct.  '''
        n, shn = 5, 4
        for i in range(n):
            lst = [np.random.randn(shn)for _ in range(n)]
            lst[i] = None
            M = vblock(lst)
            for i1 in range(n):
                B = M[shn*i1:shn*(i1+1)]
                if i1 != i: 
                    eps = np_norm(B - lst[i1])/np_norm(B)
                    self.assertTrue(eps < ZEROp)
                else:
                    eps = np.max(np.abs(B))
                    self.assertTrue(eps < ZEROp)
        for i in range(n-1):
            lst = [np.random.randn(shn)for _ in range(n)]
            lst[i] = None
            lst[i+1] = None
            M = vblock(lst)
            for i1 in range(n):
                B = M[shn*i1:shn*(i1+1)]
                if i1 != i and i1 != i+1: 
                    eps = np_norm(B - lst[i1])/np_norm(B)
                    self.assertTrue(eps < ZEROp)
                else:
                    eps = np.max(np.abs(B))
                    self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt_1(self):
        ''' In the both modes the results should be the same.  '''
        res1 = vblock([self.x, self.y, self.z])
        res2 = vblock([self.x_tt, self.y_tt, self.z_tt], self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)  
        
    def test_np_vs_tt_2(self):
        ''' In the both modes the results should be the same.  '''
        res1 = vblock([self.x, self.y, None, self.z])
        res2 = vblock([self.x_tt, self.y_tt, 2, self.z_tt], self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)  
        
    def test_np_vs_tt_3(self):
        ''' In the both modes the results should be the same.  '''
        res1 = vblock([self.x, 2, self.y, None, self.z])
        res2 = vblock([self.x_tt, None, self.y_tt, 2, self.z_tt], self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau) 
        
class TestUtilsBlockAndSpace_kronm(TestUtilsBlockAndSpace):
    ''' 
    Tests for function kronm
    from module utils.block_and_space.
    '''
        
    def test_np_vector_is_correct(self):
        ''' For MODE_NP it should be equal x \kron y \kron z  '''
        res1 = kronm([self.x, self.y, self.z], self.tau)
        res2 = np.kron(np.kron(self.x, self.y), self.z)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_matrix_is_correct(self):
        ''' For MODE_NP it should be equal A \kron B \kron C  '''
        res1 = kronm([self.A, self.B, self.C], self.tau)
        res2 = np.kron(np.kron(self.A, self.B), self.C)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_incorrect_list(self):
        ''' If list has entries of different modes, 
            the function should raise error  '''
        raised = False
        try:
            kronm([self.x, self.y, self.z_tt], self.tau)
        except:
            raised = True
        self.assertTrue(raised)
        
    def test_np_vs_tt_vector(self):
        ''' In the both modes the results should be the same for vectors.  '''
        res1 = kronm([self.x, self.y, self.z])
        res2 = kronm([self.x_tt, self.y_tt, self.z_tt], self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        res1 = kronm([self.A, self.B, self.C])
        res2 = kronm([self.A_tt, self.B_tt, self.C_tt], self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)   
        
    def test_np_vs_sp_matrix(self):
        ''' In the both modes the results should be the same for matrices.  '''
        A1 = sp_diag([self.x], [0], format='csr')
        A2 = sp_diag([self.y], [0], format='csr')
        res1 = kronm([A1.toarray(), A2.toarray()])
        res2 = kronm([A1, A2]).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
      
class TestUtilsBlockAndSpace_space_kron(TestUtilsBlockAndSpace):
    ''' 
    Tests for function space_kron
    from module utils.block_and_space.
    '''
    
    def test_np_incorrect_dim(self):
        ''' For incorrect dimension function should raise error.  '''
        raised = False
        try:
            space_kron(self.x, 1, self.d, self.n, dim=0, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        
    def test_np_incorrect_axis(self):
        ''' For incorrect axis number function should raise error.  '''
        raised = False
        try:
            space_kron(self.x, 1, self.d, self.n, dim=1, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        raised = False
        try:
            space_kron(self.x, 2, self.d, self.n, dim=1, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        raised = False
        try:
            space_kron(self.x, 2, self.d, self.n, dim=2, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
  
    def test_np_is_correct_1d_vector(self):
        ''' For 1D case the result should be correct  '''
        res1 = space_kron(self.x, 0, self.d, self.n, dim=1)
        res2 = self.x
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_is_correct_2d_vector(self):
        ''' For 2D case the result should be correct  '''
        res1 = space_kron(self.x, 0, self.d, self.n, dim=2)
        res2 = np.kron(np.ones(self.x.shape), self.x)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        res1 = space_kron(self.x, 1, self.d, self.n, dim=2)
        res2 = np.kron(self.x, np.ones(self.x.shape))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_is_correct_3d_vector(self):
        ''' For 3D case the result should be correct  '''
        res1 = space_kron(self.x, 0, self.d, self.n, dim=3)
        res2 = np.kron(np.ones(self.x.shape), np.kron(np.ones(self.x.shape), self.x))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        res1 = space_kron(self.x, 1, self.d, self.n, dim=3)
        res2 = np.kron(np.ones(self.x.shape), np.kron(self.x, np.ones(self.x.shape)))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        res1 = space_kron(self.x, 2, self.d, self.n, dim=3)
        res2 = np.kron(self.x, np.kron(np.ones(self.x.shape), np.ones(self.x.shape)))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_is_correct_1d_matrix(self):
        ''' For 1D case the result should be correct  '''
        res1 = space_kron(self.A, 0, self.d, self.n, dim=1)
        res2 = self.A
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_is_correct_2d_matrix(self):
        ''' For 2D case the result should be correct  '''
        res1 = space_kron(self.A, 0, self.d, self.n, dim=2)
        res2 = np.kron(np.eye(self.A.shape[0]), self.A)
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        res1 = space_kron(self.A, 1, self.d, self.n, dim=2)
        res2 = np.kron(self.A, np.eye(self.A.shape[0]))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_is_correct_3d_matrix(self):
        ''' For 3D case the result should be correct  '''
        res1 = space_kron(self.A, 0, self.d, self.n, dim=3)
        res2 = np.kron(np.eye(self.A.shape[0]), np.kron(np.eye(self.A.shape[0]), self.A))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        res1 = space_kron(self.A, 1, self.d, self.n, dim=3)
        res2 = np.kron(np.eye(self.A.shape[0]), np.kron(self.A, np.eye(self.A.shape[0])))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        res1 = space_kron(self.A, 2, self.d, self.n, dim=3)
        res2 = np.kron(self.A, np.kron(np.eye(self.A.shape[0]), np.eye(self.A.shape[0])))
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < ZEROp)
        
    def test_np_vs_tt_1d_vector(self):
        ''' In the both modes the results should be the same for 1D case.  '''
        res1 = space_kron(self.x, 0, self.d, self.n, dim=1)
        res2 = space_kron(self.x_tt, 0, self.d, self.n, dim=1, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)

    def test_np_vs_tt_2d_vector(self):
        ''' In the both modes the results should be the same for 2D case.  '''
        res1 = space_kron(self.x, 0, self.d, self.n, dim=2)
        res2 = space_kron(self.x_tt, 0, self.d, self.n, dim=2, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.x, 1, self.d, self.n, dim=2)
        res2 = space_kron(self.x_tt, 1, self.d, self.n, dim=2, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_3d_vector(self):
        ''' In the both modes the results should be the same for 3D case.  '''
        res1 = space_kron(self.x, 0, self.d, self.n, dim=3)
        res2 = space_kron(self.x_tt, 0, self.d, self.n, dim=3, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.x, 1, self.d, self.n, dim=3)
        res2 = space_kron(self.x_tt, 1, self.d, self.n, dim=3, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.x, 2, self.d, self.n, dim=3)
        res2 = space_kron(self.x_tt, 2, self.d, self.n, dim=3, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_1d_matrix(self):
        ''' In the both modes the results should be the same for 1D case.  '''
        res1 = space_kron(self.A, 0, self.d, self.n, dim=1)
        res2 = space_kron(self.A_tt, 0, self.d, self.n, dim=1, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)

    def test_np_vs_tt_2d_matrix(self):
        ''' In the both modes the results should be the same for 2D case.  '''
        res1 = space_kron(self.A, 0, self.d, self.n, dim=2)
        res2 = space_kron(self.A_tt, 0, self.d, self.n, dim=2, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.A, 1, self.d, self.n, dim=2)
        res2 = space_kron(self.A_tt, 1, self.d, self.n, dim=2, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_3d_matrix(self):
        ''' In the both modes the results should be the same for 3D case.  '''
        res1 = space_kron(self.A, 0, self.d, self.n, dim=3)
        res2 = space_kron(self.A_tt, 0, self.d, self.n, dim=3, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.A, 1, self.d, self.n, dim=3)
        res2 = space_kron(self.A_tt, 1, self.d, self.n, dim=3, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.A, 2, self.d, self.n, dim=3)
        res2 = space_kron(self.A_tt, 2, self.d, self.n, dim=3, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_sp_1d_matrix(self):
        ''' In the both modes the results should be the same for 1D case.  '''
        res1 = space_kron(self.I, 0, self.d, self.n, dim=1)
        res2 = space_kron(self.I_sp, 0, self.d, self.n, dim=1, tau=self.tau).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)

    def test_np_vs_sp_2d_matrix(self):
        ''' In the both modes the results should be the same for 2D case.  '''
        res1 = space_kron(self.I, 0, self.d, self.n, dim=2)
        res2 = space_kron(self.I_sp, 0, self.d, self.n, dim=2, tau=self.tau).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.I, 1, self.d, self.n, dim=2)
        res2 = space_kron(self.I_sp, 1, self.d, self.n, dim=2, tau=self.tau).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_sp_3d_matrix(self):
        ''' In the both modes the results should be the same for 3D case.  '''
        res1 = space_kron(self.I, 0, self.d, self.n, dim=3)
        res2 = space_kron(self.I_sp, 0, self.d, self.n, dim=3, tau=self.tau).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.I, 1, self.d, self.n, dim=3)
        res2 = space_kron(self.I_sp, 1, self.d, self.n, dim=3, tau=self.tau).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = space_kron(self.I, 2, self.d, self.n, dim=3)
        res2 = space_kron(self.I_sp, 2, self.d, self.n, dim=3, tau=self.tau).toarray()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
class TestUtilsBlockAndSpace_sum_out(TestUtilsBlockAndSpace):
    ''' 
    Tests for function sum_out
    from module utils.block_and_space.
    '''
        
    def test_np_incorrect_dim(self):
        ''' For incorrect dimension function should raise error.  '''
        raised = False
        try:
            sum_out(self.x, 1, self.d, self.n, dim=0, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        
    def test_np_incorrect_axis(self):
        ''' For incorrect axis number function should raise error.  '''
        raised = False
        try:
            sum_out(self.x, 1, self.d, self.n, dim=1, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        raised = False
        try:
            sum_out(self.x, 2, self.d, self.n, dim=1, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        raised = False
        try:
            kron_out(np.arange(self.n*self.n), 2, self.d, self.n, dim=2, tau=None)
        except:
            raised = True
        self.assertTrue(raised)

    def test_np_trivial_1d(self):
        ''' For 1D case it's do nothing.  '''
        res1 = sum_out(self.x, 0, self.d, self.n, dim=1)
        res2 = self.x
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)

    def test_np_vs_tt_2d(self):
        ''' In the both modes the results should be the same for 2D case.  '''
        d = 3; n = 2**d
        x = np.arange(n*n)+2.; x[2] = -4.
        x_tt = tt.tensor(x.reshape([2]*(2*d), order='F'), self.tau)
        
        res1 = sum_out(x, 0, d, n, dim=2)
        res2 = sum_out(x_tt, 0, d, n, dim=2, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = sum_out(x, 1, d, n, dim=2)
        res2 = sum_out(x_tt, 1,d, n, dim=2, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_3d(self):
        ''' In the both modes the results should be the same for 3D case.  '''
        d = 3; n = 2**d
        x = np.arange(n*n*n)+2.; x[2] = -4.
        x_tt = tt.tensor(x.reshape([2]*(3*d), order='F'), self.tau)
        
        res1 = sum_out(x, 0, d, n, dim=3)
        res2 = sum_out(x_tt, 0, d, n, dim=3, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = sum_out(x, 1, d, n, dim=3)
        res2 = sum_out(x_tt, 1, d, n, dim=3, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = sum_out(x, 2, d, n, dim=3)
        res2 = sum_out(x_tt, 2, d, n, dim=3, tau=self.tau).full().flatten('F')
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
class TestUtilsBlockAndSpace_kron_out(TestUtilsBlockAndSpace):
    ''' 
    Tests for function kron_out
    from module utils.block_and_space.
    '''
        
    def test_np_incorrect_dim(self):
        ''' For incorrect dimension function should raise error.  '''
        raised = False
        try:
            kron_out(self.x, 1, self.d, self.n, dim=0, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        
    def test_np_incorrect_axis(self):
        ''' For incorrect axis number function should raise error.  '''
        raised = False
        try:
            kron_out(self.x, 1, self.d, self.n, dim=1, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        raised = False
        try:
            kron_out(self.x, 2, self.d, self.n, dim=1, tau=None)
        except:
            raised = True
        self.assertTrue(raised) 
        raised = False
        try:
            kron_out(self.x, 2, self.d, self.n, dim=2, tau=None)
        except:
            raised = True
        self.assertTrue(raised)

    def test_np_trivial_1d(self):
        ''' For 1D case it's do nothing.  '''
        res1 = kron_out(self.x, 0, self.d, self.n, dim=1)
        res2 = self.x
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)

    def test_np_vs_tt_2d(self):
        ''' In the both modes the results should be the same for 2D case.  '''
        res1 = kron_out(self.x, 0, self.d, self.n, dim=2)
        res2 = kron_out(self.x_tt, 0, self.d, self.n, dim=2, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = kron_out(self.x, 1, self.d, self.n, dim=2)
        res2 = kron_out(self.x_tt, 1, self.d, self.n, dim=2, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
    def test_np_vs_tt_3d(self):
        ''' In the both modes the results should be the same for 3D case.  '''
        d = 3; n = 2**d
        x = np.arange(n*n)+2.; x[2] = -4.
        x_tt = tt.tensor(x.reshape([2]*(2*d), order='F'), self.tau)
        
        res1 = kron_out(x, 0, d, n, dim=3)
        res2 = kron_out(x_tt, 0, d, n, dim=3, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = kron_out(x, 1, d, n, dim=3)
        res2 = kron_out(x_tt, 1, d, n, dim=3, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        res1 = kron_out(x, 2, d, n, dim=3)
        res2 = kron_out(x_tt, 2, d, n, dim=3, tau=self.tau).full()
        eps = np_norm(res1 - res2)/np_norm(res1)
        self.assertTrue(eps < self.tau)
        
if __name__ == '__main__':
    unittest.main()