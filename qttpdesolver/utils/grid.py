# -*- coding: utf-8 -*- 
import numpy as np

import tt
import tt.multifuncrs2 as multifuncrs2

from general import MODE_NP, MODE_SP, ttround, msum
from block_and_space import kronm, space_kron
from spec_matr import vzeros_except_one

def _mesh_cc(d, dim, tau=None, mode=MODE_NP):
    ''' 
    Generate spatial mesh on cell centers
    (x,y,z=h/2, ..., x,y,z=3h/2, ..., x,y,z=1-h/2).
    '''
    n = 2**d
    h = 1./n
    if mode == MODE_NP or mode == MODE_SP:
        gr = h * (np.arange(n) + 0.5)
        return np.meshgrid(*[gr]*dim, indexing='ij')
    gr = h * (tt.xfun(2, d) + 0.5 * tt.ones(2, d))
    return [space_kron(gr, i, d, n, dim, tau) for i in range(dim)]
    
def _mesh_lc(d, dim, tau=None, mode=MODE_NP):
    ''' 
    Generate spatial mesh on left corners
    (x,y,z=0, ..., x,y,z=h, ..., x,y,z=1-h).
    '''
    n = 2**d
    h = 1./n
    if mode == MODE_NP or mode == MODE_SP:
        gr = h * np.arange(n)
        return np.meshgrid(*[gr]*dim, indexing='ij')
    gr = h * tt.xfun(2, d)
    return [space_kron(gr, i, d, n, dim, tau) for i in range(dim)]
    
def _mesh_rc(d, dim, tau=None, mode=MODE_NP):
    ''' 
    Generate spatial mesh on right corners
    (x,y,z=h, ..., x,y,z=2h, ..., x,y,z=1).
    '''
    n = 2**d
    h = 1./n
    if mode == MODE_NP or mode == MODE_SP:
        gr = h * np.arange(1, n+1)
        return np.meshgrid(*[gr]*dim, indexing='ij')
    gr = h * (tt.xfun(2, d) + tt.ones(2, d))
    return [space_kron(gr, i, d, n, dim, tau) for i in range(dim)]
    
def _mesh_uxe(d, dim, tau=None, mode=MODE_NP):
    ''' 
    Generate spatial mesh on upper-x edge midpoints
    (x[0]=h/2, y[0]=h, z[0]=h, ..., x[-1]=1.-h/2, y[-1]=1., z[-1]=1.)
    '''
    if not dim in [1, 2, 3]:
        raise ValueError('May be constructed only for dim=1,2,3.')
    gr_c = _mesh_cc(d, dim, tau, mode)
    gr_e = _mesh_rc(d, dim, tau, mode)
    if dim==1:
        return [gr_c[0]]
    if dim==2:
        return [gr_c[0], gr_e[1]]
    if dim==3:
        return [gr_c[0], gr_e[1], gr_e[2]]
    
def _mesh_uye(d, dim, tau=None, mode=MODE_NP):
    ''' 
    Generate spatial mesh on upper-y edge midpoints
    (x[0]=h, y[0]=h/2, z[0]=h, ..., x[-1]=1., y[-1]=1-h/2,. z[-1]=1.)
    '''
    if not dim in [2, 3]:
        raise ValueError('May be constructed only for dim=2,3.')
    gr_c = _mesh_cc(d, dim, tau, mode)
    gr_e = _mesh_rc(d, dim, tau, mode)
    if dim==2:
        return [gr_e[0], gr_c[1]]
    if dim==3:
        return [gr_e[0], gr_c[1], gr_e[2]]
                
def _mesh_uze(d, dim, tau=None, mode=MODE_NP):
    ''' 
    Generate spatial mesh on upper-z edge midpoints
    (x[0]=h, y[0]=h, z[0]=h/2, ..., x[-1]=1.,, y[-1]=1., z[-1]=1.-h/2)
    '''
    if dim != 3:
        raise ValueError('May be constructed only for dim=3.')
    gr_c = _mesh_cc(d, dim, tau, mode)
    gr_e = _mesh_rc(d, dim, tau, mode)
    if dim==3:
        return [gr_e[0], gr_e[1], gr_c[2]]

def _construct_mesh(d, dim, tau=None, mode=MODE_NP, grid='cc'):
    if grid.lower() in ['cc', 'cell centers']:
        mesh = _mesh_cc(d, dim, tau, mode)
    elif grid.lower() in ['lc', 'left corners']:
        mesh = _mesh_lc(d, dim, tau, mode)
    elif grid.lower() in ['rc', 'right corners']:
        mesh = _mesh_rc(d, dim, tau, mode) 
    elif grid.lower() in ['uxe', 'upper-x edge midpoints']:
        mesh = _mesh_uxe(d, dim, tau, mode)
    elif grid.lower() in ['uye', 'upper-y edge midpoints']:
        mesh = _mesh_uye(d, dim, tau, mode)
    elif grid.lower() in ['uze', 'upper-z edge midpoints']:
        mesh = _mesh_uze(d, dim, tau, mode)
    else:
        raise ValueError('Unknown grid type.')
    return mesh
                  
def quan_on_grid(func, d, dim, tau=None, eps=None, mode=MODE_NP, grid='cc',
                 name='Unknown spatial function', verb=False, inv=False):
    '''
    Construct physical quantity on spatial mesh.
        func   - is a function (x), (x, y) or (x, y, z) that calculate quantity
        d      - grid factor (total number of cells 2^{d*dim})
        dim    - dimension (1, 2, 3)
        tau    - tolerance for TT-rounding
        eps    - tolerance for TT-cross
        mode   - is the mode of calculation (MODE_NP or MODE_TT)
        grid   - is the type of the grid:
                   'cc'  or 'cell centers'
                   'lc'  or 'left corners'  (cell corner with minimum coord.)
                   'rc'  or 'right corners' (cell corner with maximum coord.)
                   'uxe' or 'upper-x edge midpoints'
                   'uye' or 'upper-y edge midpoints'
                   'uze' or 'upper-z edge midpoints'
        name   - string for print
        verb   - if is True, then calculation proccess will be presented
        inv    - if is True, then 1/func will be constructed
    '''
    mesh = _construct_mesh(d, dim, tau, mode, grid)
    if mode == MODE_NP or mode == MODE_SP:
        x = func(*mesh).flatten('F')
        if inv:
            x = 1./x
    else:
        if verb:
            print '  Construction of %s'%name
        if not inv:
            x = multifuncrs2(mesh, func, eps, verb=verb, y0=mesh[0])
        else:
            x = multifuncrs2(mesh, lambda x: 1./func(x), eps, verb=verb, y0=mesh[0])
    return ttround(x, tau)
    
def deltas_on_grid(r_list, val_list, d, tau=None, mode=MODE_NP, grid='rc'):
    '''
    Construct a sum of delta functions on spatial grid.
        r_list   - is a list of source locations according to dimension (dim)
        val_list - is a list of values of sources
        d        - grid factor (total number of cells 2^{d*dim})
        tau      - tolerance for TT-rounding
        mode     - is the mode of calculation (MODE_NP or MODE_TT, or MODE_SP)
        grid     - is the type of the grid:
                     'rc'  or 'right corners' (cell corner with maximum coord.)
    '''     
    for i in range(len(val_list)):
        res_curr = delta_on_grid(r_list[i], val_list[i], d, tau, mode, 'rc')
        if i==0:
            res = res_curr
        else:
            res = msum([res, res_curr], tau)
    return res

def delta_on_grid(r, val, d, tau=None, mode=MODE_NP, grid='rc'):
    '''
    Construct delta function on spatial grid.
        r      - is a list of source coordinates according to dimension (dim)
        val    - is a value of source (float)
        d      - grid factor (total number of cells 2^{d*dim})
        tau    - tolerance for TT-rounding
        mode   - is the mode of calculation (MODE_NP or MODE_TT, or MODE_SP)
        grid     - is the type of the grid:
                     'rc'  or 'right corners' (cell corner with maximum coord.)
    '''
    if grid.lower()!='rc' and grid.lower()!='right corners':
        raise ValueError('Delta on grid works only for rc mesh type.')
    n = 2**d
    L = 1.
    h = L/n
    dim = len(r)   
    res = []
    for i in range(dim-1, -1, -1):
        i1, h1 = nearest_left_node(r[i], n, L)
        if i1>= n-1:
            raise ValueError('Incorrect point source location.')
        i2, h2 = i1+1, h-h1
        res.append(msum([vzeros_except_one(d, i1, mode, val*h2/h**2), 
                         vzeros_except_one(d, i2, mode, val*h1/h**2)], tau))
    return kronm(res, tau)
    
def mind2ind(mind, n):
    '''
    Construct flatten index from multi-index
        Input:
    mind   - is a multi index of length dim
    n      - total number of grid cells along one axis
        Output:
    ind    - integer index >=0, <=n**dim-1
    '''
    dim = len(mind)
    ind = mind[0]
    for i in range(1, dim):
        ind+= mind[i]*(n**i)
    return ind
 
def nearest_left_node(x, n, L, grid='rc'):
    '''
    Get left nearest node of 1D spatial grid to a given point x and distance.
        Input:
    x      - is a coordinate of the point
    n      - total number of grid cells
    L      - size of the rectangular grid
    grid   - is the type of the grid:
               'rc'  or 'right corners' (cell corner with maximum coord.)
        Output:
    i1, h1  - left nearest index and distance
    '''
    if grid.lower()!='rc' and grid.lower()!='right corners':
        raise ValueError('coord2ind works only for rc grid type.')
    h = L/n
    if x<h:
        return None, None
    i1 = int(x/h)-1
    h1 = x - (h+h*i1)
    return i1, h1 
    
def coord2ind(r, d, L, grid='rc'):
    '''
    Construct nearest index on flatten (with Fortran ordering style)
    spatial grid to given coordinate r.
        Input:
    r      - is a radius-vector of the length dim
    d      - grid factor (total number of cells 2^{d*dim})
    L      - size of the rectangular grid
    grid   - is the type of the grid:
               'rc'  or 'right corners' (cell corner with maximum coord.)
        Output:
    mind   - list of length dim, that is a multi-index of the nearest grid point
    '''
    if grid.lower()!='rc' and grid.lower()!='right corners':
        raise ValueError('coord2ind works only for rc grid type.')
    n = 2**d
    h = L/n
    dim = len(r)
    mind = []
    for i in range(dim):
        ind = int(r[i]/h)
        if r[i]%h<h/2:
            ind = ind - 1
        if ind<0:
            mind.append(0)
        elif ind>=n:
            mind.append(n-1)
        else:
            mind.append(ind)
    return mind