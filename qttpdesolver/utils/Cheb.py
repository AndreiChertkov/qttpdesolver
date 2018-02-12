# -*- coding: utf-8 -*-
import time
import numpy as np
import tt
from tt.cross import rect_cross

from . import MODE_NP, MODE_TT, MODE_SP, Vector
from ..utils.capture_output import CaptureCross

class Cheb(object):

    def __init__(self):
        self.time = None
        self.A = None

    def interpolate(self, Y):
        '''
        ___DESCRIPTION
        Find coefficients A_i1...iN for interpolation of the multi dimensional
        function by Chebyshev polynomials in the form:
        f(x1,x2,...,xN) = Sum ( a_i1...iN * T_i1(x1)*T_i2(x2)*...*T_iN(xN) ).
        ___INPUT
        Y       - tensor of function values in nodes of the Chebushev mesh
                  (for different axis numbers of points may not be equal)
                  type: ndarray [dimensions] of float
        ___OUTPUT
        A       - constructed tensor of coefficients
                  type: ndarray [dimensions] of float
        '''
        A = Y.copy()
        for i in range(len(Y.shape)):
            A = np.swapaxes(A, 0, i)
            A = A.reshape((Y.shape[i], -1))
            A = ch_ut.interpolate_1d(A)
            A = A.reshape(Y.shape)
            A = np.swapaxes(A, i, 0)
        self.A = A
        return A

    def func_val(self, X, lim):
    '''
    ___DESCRIPTION
    Calculate values of interpolated function in given x points.
    ___INPUT
    X       - values of x variable
              type: ndarray [dimensions, number of points] of float
    A       - is the tensor of coefficients
              type: ndarray [dimensions] of float
    lim     - are min-max values of variable for every dimension
              type: ndarray [dimensions, 2] of float
    ___OUTPUT
    Y       - approximated values of the function in given points
              type: ndarray [number of points] of float
    '''
    Y = np.zeros(X.shape[1])
    for j in xrange(X.shape[1]):
        A_tmp = self.A.copy()
        T = (2.*X[:, j]-x_lim[:, 1]-x_lim[:, 0]) / (x_lim[:, 1]-x_lim[:, 0])
        for i in xrange(X.shape[0]):
            Tch = ch_ut.polynomial(self.A.shape[i], T)
            A_tmp = np.tensordot(A_tmp, Tch[:,i], axes=([0], [0]))
        Y[j] = A_tmp
    return Y

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def mesh_point(ind, sh, x_lim):
    '''
    ___DESCRIPTION____________________________________________________________
    Get point of Chebyshev multidimensional mesh for given indices ind.
    Points for every axis i are calculated as x = cos(ind[i]*pi/(sh[i]+1)),
    where n is a total number of points for selected axis. And then
    obtained points are scaled according given limits.
    ___INPUT__________________________________________________________________
    ind   - indices of mesh point
            type: ndarray [dim] of int
    sh    - total numbers of points for every dimension
            type: ndarray [dim] of int
    lim   - min-max values of variable for every dimension
            type: ndarray [dim, 2] of float
    ___OUTPUT_________________________________________________________________
    x     - is the mesh point
            type: ndarray [dim] of float
    '''
    t = np.cos(np.pi*ind/(sh - 1))
    x = t*(x_lim[:, 1] - x_lim[:, 0])/2. + (x_lim[:, 1] + x_lim[:, 0])/2.
    return x

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def polynomial_1d(n_max, x):
    '''
    ___DESCRIPTION____________________________________________________________
    Calculate Chebyshev polynomials of order until n_max in x point.
    ___INPUT__________________________________________________________________
    n_max  - number of polynomials (order = 0,1,...,n_max-1)
             type: int
    x      - value of x variable
             type: float
    ___OUTPUT_________________________________________________________________
    Tch    - Chebyshev polynomials
             type: ndarray [n_max] of float
    '''
    Tch = np.ones(n_max)
    if n_max == 1:
        return Tch
    Tch[1] = x
    for n in xrange(2, n_max):
        Tch[n] = 2.*x*Tch[n-1] - Tch[n-2]
    return Tch
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def polynomial(n_max, X):
    '''
    ___DESCRIPTION____________________________________________________________
    Calculate Chebyshev polynomials of until n_max in x points.
    ___INPUT__________________________________________________________________
    n_max  - number of polynomials (order = 0,1,...,n_max-1)
             type: int
    X      - values of x variable
             type: ndarray [dimensions, number of points] of float
    ___OUTPUT_________________________________________________________________
    Tch    - Chebyshev polynomials
             type: ndarray [n_max, X.shape] of float
    '''
    Tch = np.ones([n_max] + list(X.shape))
    if n_max == 1:
        return Tch
    Tch[1, :] = X.copy()
    for n in xrange(2, n_max):
        Tch[n, ] = 2.*X*Tch[n-1, ] - Tch[n-2, ]
    return Tch

def interpolate_1d(Y):
    '''
    ___DESCRIPTION____________________________________________________________
    Find coefficients A_i for interpolation of 1D functions
    by Chebyshev polynomials f(x) = Sum ( A_i * T_i(x) ) using fast fourier
    transform (can find coefficients for several functions on one call
    if all functions have equal numbers of mesh points).
    ___INPUT__________________________________________________________________
    Y       - function's values at the mesh nodes
              type: ndarray [number of points, number of functions] of float
    ___OUTPUT_________________________________________________________________
    A       - constructed matrix of coefficients
              type: ndarray [number of points, number of functions] of float
    '''
    n_poi = Y.shape[0]
    Yext = np.zeros((2*n_poi-2, Y.shape[1]))
    Yext[0:n_poi, :] = Y[:, :]
    for k in xrange(n_poi, 2*n_poi-2):
        Yext[k, :] = Y[2*n_poi-k-2, :]
    A = np.zeros(Y.shape)
    for n in xrange(Y.shape[1]):
        A[:, n] = (np.fft.fft(Yext[:, n]).real/(n_poi - 1))[0:n_poi]
        A[0, n] /= 2.
    return A
