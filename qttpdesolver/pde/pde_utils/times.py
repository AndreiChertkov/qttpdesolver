# -*- coding: utf-8 -*-
class Time(dict):
    '''
    Container for time durations of the main operations while PDE solution.
    '''
    
    def __init__(self):
        dict.__init__(self)
        self.names = ['mgen', 'sgen', 'solve', 'prepare_res',
                      'D', 'F', 'B', 'Q', 'W',  'R', 'H',
                      'A', 'rhs', 'amen', 'u_real', 'ud_calc', 'ud_real', 
                      'en_calc', 'en_real', 'uf_calc', 'uf_real', 'uu_calc', 'uu_real']
        self.clean()
                      
    def clean(self):
        for name in self.names:
            self.set(name)             
        
    def set(self, name, t=None): 
        self[name] = t

    def get_full(self):
        t_full = 0.
        if self['mgen'] is not None:
            t_full+= self['mgen']
        if self['sgen'] is not None:
            t_full+= self['sgen']
        if self['solve'] is not None:
            t_full+= self['solve'] 
        if t_full==0.:
            t_full = None
        return t_full