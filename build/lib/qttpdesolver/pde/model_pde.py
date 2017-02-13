# -*- coding: utf-8 -*-
from ..tensor_wrapper import MODE_TT
from models import set_model, get_models_txt
from txts import compose_model

class ModelPde(object):
    '''
    A simple container for PDE of the form -div(k grad u) = f.
    All PDE parameteres may be set in auto mode by model number or name 
    (from models module). Call self.present_models to see all available models.
    
                Global parameters [and default values]
                
    mode          - [MODE_NP] = 'np' (or MODE_NP) for numpy format,
                              = 'tt' (or MODE_TT) for tensor train format,
                              = 'sp' (or MODE_SP) for scipy sparse format
    out_file      - [None] path to file to print (in 'a'-append mode)
                    * '.txt' will be appended automatically if needed
    print_to_std  - [True]  if True, then present_* will print to std 
    print_to_file - [False] if True, then present_* will print to out_file             
                
                Model parameters [and default values]

    model_num  - [None] model (benchmark) number
    txt        - [None] representation of equation by a string 
    dim        - [None] dimension of the PDE
    L          - [1.] domain size

    k          - [None] pointer to outer function k(x), k(x, y) or k(x, y, z)
    f          - [None] pointer to outer function f(x), f(x, y) or f(x, y, z)
    u          - [None] pointer to outer function u(x), u(x, y) or u(x, y, z)
                 (exact analytical solution, if is known)
    ux         - [None] pointer to outer function d/dx u(x), u(x, y) or u(x, y, z)
                 (exact analytical derivative, if is known)
    uy         - [None] pointer to outer function d/dy u(x), u(x, y) or u(x, y, z)
                 (exact analytical derivative, if is known)
                 * only for 2D and 3D case
    uz         - [None] pointer to outer function d/dz u(x), u(x, y) or u(x, y, z)
                 (exact analytical derivative, if is known)
                 * only for 3D case
                 
    k_txt      - [None] representation of k function by a string
    f_txt      - [None] representation of f function by a string
    u_txt      - [None] representation of u function by a string
    ux_txt     - [None] representation of ux function by a string
    uy_txt     - [None] representation of uy function by a string
    uz_txt     - [None] representation of uz function by a string
    
    params     - [None] are params that are passed to k, f, ... functions.
                 * all funcs. k, f, u,... must have the same input parameters
    params_txt - [None] representation of params by a string with %f. 
                 Embedding params_txt%(tuple(params)) should be possible    

                 Functions

    set_dim         - set spatial dimension
    set_mode        - set mode of calculations
    set_model       - set PDE model via number or name (models.py will be used)
    set_params      - set a list of parameters for k, f, u, ... functions
    clean_model     - delete current model (set all Model parameters to None)
                      * is called from set_model automatically
    k_func          - get spatial values of the k-coefficient
                      * for MODE_NP and MODE_SP it expects input in the form
                      (x), (x, y), (x, y, z) (according to PDE dimension),
                      where x,... are float or numpy arrays, 
                      * for MODE_TT input should be like (r) with 2-d array r, 
                      where columns are (x), (x, y) or (x, y, z)
    f_func          - the same for the f-function
    u_func          - the same for the u-function
    ux_func         - the same for the ux-function
    uy_func         - the same for the uy-function
    uz_func         - the same for the uz-function
    
    present_models  - present all available PDE models
    present_model   - present current (selected) PDE model (k, f, u and so on)
    '''
    
    def __init__(self):
        self.out_file      = None
        self.print_to_std  = True
        self.print_to_file = False
        self.set_dim(None)
        self.set_mode(None)
        self.set_model(None)
        
    def set_dim(self, dim):
        self.dim = dim
        
    def set_mode(self, mode):
        self.mode = mode
        
    def set_model(self, model_selected=None):
        self.clean_model()
        if model_selected is not None:
            self.model_num = set_model(self, model_selected)

    def set_params(self, params):
        if not isinstance(params, list):
            raise ValueError('Input parameters should be in list.')
        if self.params is not None:
            if len(params) != len(self.params):
                raise ValueError('Incorrect number of parameters.')
        self.params = [p for p in params]

    def clean_model(self):
        self.txt, self.dim, self.L = None, None, 1.
        self.params_txt, self.params, self.model_num = None, None, None
        self.k_txt,  self.k  = None, None
        self.f_txt,  self.f  = None, None
        self.u_txt,  self.u  = None, None
        self.ux_txt, self.ux = None, None
        self.uy_txt, self.uy = None, None
        self.uz_txt, self.uz = None, None
        
    def _split_r(self, r):
        if not self.mode==MODE_TT:
            return list(r)
        return [r[0][:, i] for i in range(self.dim)]
    
    def k_func(self, *r):
        return self.k(*(self._split_r(r)+self.get_params_list()))
        
    def f_func(self, *r):
        return self.f(*(self._split_r(r)+self.get_params_list()))
        
    def u_func(self, *r):
        return self.u(*(self._split_r(r)+self.get_params_list()))
    
    def ux_func(self, *r):
        return self.ux(*(self._split_r(r)+self.get_params_list()))
    
    def uy_func(self, *r):
        return self.uy(*(self._split_r(r)+self.get_params_list()))
    
    def uz_func(self, *r):
        return self.uz(*(self._split_r(r)+self.get_params_list()))
            
    def get_params_list(self):
        if self.params is not None:
            return self.params
        return []

    def present_models(self):
        self._present('The following models are available:\n'+get_models_txt())
            
    def present_model(self):
        self._present(compose_model(self))
        
    def _params2str(self):
        if self.params_txt is None or self.params is None:
            return ''
        return str(self.params_txt%(tuple(self.params)))
        
    def _present(self, s):
        ''' Print given string to std and/or to file according settings.  '''
        if self.print_to_std:
            print s
        if self.print_to_file and self.out_file is not None:
            f = open(self.out_file+'.txt', "a+")
            if not s.endswith('\n'):
                s+= '\n'
            f.writelines(s)
            f.close()