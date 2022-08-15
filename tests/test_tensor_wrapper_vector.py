# -*- coding: utf-8 -*-
import unittest
import numpy as np

from qttpdesolver import MODE_NP, MODE_TT, MODE_SP
from qttpdesolver import Vector

class Test_tensor_wrapper_vector(unittest.TestCase):
    ''' Base class for 1D diffusion problem  '''

    def setUp(self):
        self.d = 4
        self.mode = MODE_NP
        self.tau = 1.E-8
        self.i = 3
        self.val = 2.
        self.name = 'VECTOR-TEST'

    def assertEqual(self, a, b, tau=1.E-8):
        self.assertTrue(np.abs(a-b) < tau)

class Test_tensor_wrapper_vector_base(Test_tensor_wrapper_vector):

    def test_name(self):
        v = Vector(None, self.d, self.mode, self.tau, False, self.name)
        self.assertTrue(v.name == self.name)

class Test_tensor_wrapper_vector_diag(Test_tensor_wrapper_vector):

    def test_form_np(self):
        v = Vector.unit(self.d, MODE_NP, self.tau, 1, 4.)
        v+= Vector.arange(self.d, MODE_NP, self.tau)
        M = v.diag()
        self.assertEqual(v.rel_err(M.diag()), 0.)
        for i in range(2**self.d):
            for j in range(2**self.d):
                if i==j:
                    self.assertEqual(M.x[i, j], v.x[i])
                else:
                    self.assertEqual(M.x[i, j], 0.)

    def test_form_tt(self):
        v = Vector.unit(self.d, MODE_TT, self.tau, 1, 4.)
        v+= Vector.arange(self.d, MODE_TT, self.tau)
        M = v.diag()
        self.assertEqual(v.rel_err(M.diag()), 0.)

class Test_tensor_wrapper_vector_sum(Test_tensor_wrapper_vector):

    def test_x(self):
        v1 = Vector.unit(self.d, self.mode, self.tau, 1, 4.)
        v2 = Vector.unit(self.d+1, self.mode, self.tau, 2, 5.1)
        try:
            v = v1 + v2
        except:
            raised = True
        self.assertTrue(raised)

class Test_tensor_wrapper_vector_unit(Test_tensor_wrapper_vector):

    def test_base(self):
        v = Vector.unit(self.d, MODE_NP, self.tau, 1, 5., self.name)
        self.assertTrue(v.d == self.d)
        self.assertTrue(v.mode == MODE_NP)
        self.assertEqual(v.tau, self.tau)
        self.assertTrue(v.name == self.name)

    def test_form_np(self):
        i0, i1, val = 1, -2, 5.
        v = Vector.unit(self.d, MODE_NP, self.tau, i0, val)
        v+= Vector.unit(self.d, MODE_NP, self.tau, i1, val)
        self.assertEqual(v.sum(), 2*val)
        for i in range(2**self.d):
            if i in [i0, 2**self.d+i1]:
                self.assertEqual(v.x[i], val)
            else:
                self.assertEqual(v.x[i], 0.)

    def test_form_tt(self):
        i0, i1, val = 1, -2, 5.
        v_np = Vector.unit(self.d, MODE_NP, self.tau, i0, val)
        v_np+= Vector.unit(self.d, MODE_NP, self.tau, i1, val)
        v = Vector.unit(self.d, MODE_TT, self.tau, i0, val)
        v+= Vector.unit(self.d, MODE_TT, self.tau, i1, val)
        self.assertEqual(v.sum(), 2*val)
        v.conv2mode(MODE_NP)
        self.assertTrue(v.rel_err(v_np) < self.tau)

class Test_tensor_wrapper_vector_ones(Test_tensor_wrapper_vector):

    def test_base(self):
        v = Vector.ones(self.d, MODE_NP, self.tau, self.name)
        self.assertTrue(v.d == self.d)
        self.assertTrue(v.mode == MODE_NP)
        self.assertEqual(v.tau, self.tau)
        self.assertTrue(v.name == self.name)

    def test_form_np(self):
        v = Vector.ones(self.d, MODE_NP, self.tau)
        self.assertEqual(v.sum(), 2**self.d)
        for i in range(2**self.d):
            self.assertEqual(v.x[i], 1.)

    def test_form_tt(self):
        v_np = Vector.ones(self.d, MODE_NP, self.tau)
        v = Vector.ones(self.d, MODE_TT, self.tau)
        self.assertEqual(v.sum(), 2**self.d)
        v.conv2mode(MODE_NP)
        self.assertTrue(v.rel_err(v_np) < self.tau)

class Test_tensor_wrapper_vector_arange(Test_tensor_wrapper_vector):

    def test_base(self):
        v = Vector.arange(self.d, MODE_NP, self.tau, self.name)
        self.assertTrue(v.d == self.d)
        self.assertTrue(v.mode == MODE_NP)
        self.assertEqual(v.tau, self.tau)
        self.assertTrue(v.name == self.name)

    def test_form_np(self):
        v = Vector.arange(self.d, MODE_NP, self.tau)
        self.assertEqual(v.sum(), (2**self.d - 1) * 2**self.d/2)
        for i in range(2**self.d):
            self.assertEqual(v.x[i], i)

    def test_form_tt(self):
        v_np = Vector.arange(self.d, MODE_NP, self.tau)
        v = Vector.arange(self.d, MODE_TT, self.tau)
        self.assertEqual(v.sum(), (2**self.d - 1) * 2**self.d/2)
        v.conv2mode(MODE_NP)
        self.assertTrue(v.rel_err(v_np) < self.tau)

class Test_tensor_wrapper_vector_block(Test_tensor_wrapper_vector):

    def test_base(self):
        vlist = []
        vlist.append(Vector.unit(self.d, MODE_NP, self.tau, 1, 5.))
        vlist.append(Vector.unit(self.d, MODE_NP, self.tau,-1, 5.))
        vlist.append(Vector.ones(self.d, MODE_NP, self.tau))
        vlist.append(Vector.arange(self.d, MODE_NP, self.tau))
        v = Vector.block(vlist, self.name)
        self.assertTrue(v.d == 2+self.d)
        self.assertTrue(v.mode == MODE_NP)
        self.assertEqual(v.tau, self.tau)
        self.assertTrue(v.name == self.name)
        try:
            v = Vector.block(vlist[0:3], self.name)
        except:
            raised = True
        self.assertTrue(raised)
        try:
            v = Vector.block(Vector.unit(4, MODE_NP), self.name)
        except:
            raised = True
        self.assertTrue(raised)

    def test_form_np(self):
        vlist = []
        vlist.append(Vector.unit(self.d, MODE_NP, self.tau, 1, 5.))
        vlist.append(Vector.unit(self.d, MODE_NP, self.tau,-1, 5.))
        vlist.append(Vector.ones(self.d, MODE_NP, self.tau))
        vlist.append(Vector.arange(self.d, MODE_NP, self.tau))
        v = Vector.block(vlist)
        n = 2**self.d
        s = 5. + 5. + n + (n - 1) * n/2
        self.assertEqual(v.sum(), s)
        for i in range(4*n):
            self.assertEqual(v.x[i], vlist[i/n].x[i-(i/n)*n])

    def test_zero_np(self):
        vlist = []
        vlist.append(None)
        vlist.append(Vector.unit(self.d, MODE_NP, self.tau,-1, 5.))
        vlist.append(Vector.ones(self.d, MODE_NP, self.tau))
        vlist.append(Vector.arange(self.d, MODE_NP, self.tau))
        v = Vector.block(vlist)
        n = 2**self.d
        for i in range(n):
            self.assertEqual(v.x[i], 0.)

    def test_form_tt(self):
        vlist = []
        vlist.append(Vector.unit(self.d, MODE_TT, self.tau, 1, 5.))
        vlist.append(Vector.unit(self.d, MODE_TT, self.tau,-1, 5.))
        vlist.append(Vector.ones(self.d, MODE_TT, self.tau))
        vlist.append(Vector.arange(self.d, MODE_TT, self.tau))
        v = Vector.block(vlist, self.name)
        n = 2**self.d
        s = 5. + 5. + n + (n - 1) * n/2
        self.assertEqual(v.sum(), s)
        for vl in vlist:
            vl.conv2mode(MODE_NP)
        v_np = Vector.block(vlist)
        v.conv2mode(MODE_NP)
        self.assertTrue(v.rel_err(v_np) < self.tau)

if __name__ == '__main__':
    unittest.main()
