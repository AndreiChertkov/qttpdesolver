# -*- coding: utf-8 -*-
from ..tensor_wrapper.tensor_base import MODE_NP
from ..tensor_wrapper.vector import Vector

class Grid(object):
    
    def __init__(self):
        self.set_params(d=None, h=None, dim=None)
        self.clean()
        
    def set_params(self, d, h, dim, tau=None, mode=MODE_NP):
        self.d = d
        self.h = h
        self.dim = dim
        self.tau = tau
        self.mode = mode
            
    def clean(self):
        self.xc, self.yc, self.zc = None, None, None
        self.xr, self.yr, self.zr = None, None, None
        
    def construct(self):
        ''' 
        Generate spatial mesh on cell centers
        (xc,yc,zc=h/2, ..., xc,yc,zc=3h/2, ..., xc,yc,zc=L-h/2).
        and spatial mesh on right corners
        (xr,yr,zr=h  , ..., xr,yr,zr=2h  , ..., xr,yr,zr=L).
        '''
        a = Vector.arange(self.d, self.mode, self.tau)
        e = Vector.ones(self.d, self.mode, self.tau)
        r_center = self.h * (a + e*0.5)
        r_right  = self.h * (a + e) 
        
        self.clean()
        if self.dim==1:  
            self.xc = r_center
            self.xr = r_right
        if self.dim==2: 
            self.xc = e.kron(r_center)    
            self.yc = r_center.kron(e)
            self.xr = e.kron(r_right)    
            self.yr = r_right.kron(e)
        if self.dim==3:        
            self.xc = e.kron(e).kron(r_center) 
            self.yc = e.kron(r_center).kron(e)   
            self.zc = r_center.kron(e).kron(e) 
            self.xr = e.kron(e).kron(r_right) 
            self.yr = e.kron(r_right).kron(e)   
            self.zr = r_right.kron(e).kron(e)