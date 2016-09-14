# -*- coding: utf-8 -*-

class AlgssPar(dict):
    
    def __init__(self):
        dict.__init__(self)
        self.names_in  = ['nswp', 'kickrank', 'local_prec', 'local_iters', 
                          'local_restart', 'trunc_norm', 'max_full_size',
                          'tau', 'tau_u_calc_from_algss']
        self.names_out = ['iters', 'max_dx', 'max_res', 'max_rank']
        self.clean_in()
        self.clean_out()
        
    def clean_in(self):
        for name in self.names_in:
            self.set(name)             
        
    def clean_out(self):
        for name in self.names_out:
            self.set(name) 
            
    def set(self, name, val=None): 
        self[name] = val
        
    def copy(self):
        AlgssPar_tmp = AlgssPar()
        for name in self.names_in+self.names_out:
            AlgssPar_tmp.set(name, self[name])
        return AlgssPar_tmp