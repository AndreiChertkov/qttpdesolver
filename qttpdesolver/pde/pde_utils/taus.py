# -*- coding: utf-8 -*-
class Tau(dict):
    '''
    Container for tolerances of calculations in the TT-format (MODE_TT).
    '''
    
    def __init__(self):
        dict.__init__(self)
        self.names = ['mgen', 'sgen', 'solve', 'u_calc' , 'u_real', 
                      'ud_calc', 'ud_real', 'en_calc', 'en_real']
        self.txt = {}
        self.txt['mgen']    = 'Matrices (iD, Q, W,...) construction'
        self.txt['sgen']    = 'Matrix A and vector rhs construction'
        self.txt['solve']   = 'Solution of the system A w_x = rhs and splitting of block system solution' 
        self.txt['u_calc']  = 'Rounding of calculated solution' 
        self.txt['u_real']  = 'Construction of analytical solution'
        self.txt['ud_calc'] = 'Construction of derivative from obtained PDE solution'
        self.txt['ud_real'] = 'Construction of analytical derivative'     
        self.txt['en_calc'] = 'Construction of energy from derivative of PDE solution'
        self.txt['en_real'] = 'Construction of energy from obtained analytical derivative'
        self.clean()
                      
    def clean(self):
        for name in self.names:
            self.set(name)             
        
    def set(self, name, tau_round=None, tau_cross=None, tau_amens=None): 
        self[name] = {'round':tau_round, 'cross':tau_cross, 'amens':tau_amens}
        
    def set_by_main(self, tau_round, tau_cross, tau_amens):
        self.set('mgen',    tau_round, tau_cross, None)
        self.set('sgen',    self['mgen']['round'], None, None)
        self.set('solve',   tau_amens, None, tau_amens)
        self.set('u_calc',  tau_amens, None, None)
        self.set('u_real',  self['u_calc']['round']/100., 
                            self['u_calc']['round']/100., None)
        self.set('ud_calc', self['u_calc']['round'],
                            self['u_calc']['round'], None)
        self.set('ud_real', self['ud_calc']['round']/100.,
                            self['ud_calc']['cross']/100., None)
        self.set('en_calc', self['ud_calc']['round'],
                            self['ud_calc']['cross'], None)
        self.set('en_real', self['en_calc']['round']/100.,
                            self['en_calc']['cross']/100., None)