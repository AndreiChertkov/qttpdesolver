# -*- coding: utf-8 -*- 
from ...utils.general import MODE_NP, MODE_TT, MODE_SP
  
def compose_model(PDE):
    ''' Prepare string representation of the PDE model.  '''
    s = ''
    s+= _str('PDE:        ', PDE.txt)
    s+= _str('Parameters: ', PDE._params2str())
    s+= _str('', PDE.k_txt)
    s+= _str('', PDE.f_txt)
    s+= _str('', PDE.u_txt)
    s+= _str('', PDE.ux_txt)
    s+= _str('', PDE.uy_txt)
    s+= _str('', PDE.uz_txt)
    return s
        
def compose_res_1s(PDE):
    ''' Prepare one-string representation of the computation result.  '''
    ud_err = [None]*3
    if isinstance(PDE.ud_err, list):
        for i in range(len(PDE.ud_err)):
            ud_err[i] = PDE.ud_err[i]
    s = ''
    s+= _str('d=',       PDE.d         ,           '%-2d'  , '')
    s+= _str('|',        PDE.solver_txt,           '%-2s'  , '')
    s+= _str('-',        _mode2str(PDE.mode),      '%-2s'  , '')
    s+= _str('|er=',     PDE.u_err,                '%-8.1e', '')
    s+= _str('|erdx=',   ud_err[0],                '%-8.1e', '')
    s+= _str('|erdy=',   ud_err[1],                '%-8.1e', '')
    s+= _str('|erdz=',   ud_err[2],                '%-8.1e', '')
    s+= _str('|ere=',    PDE.en_err,               '%-8.1e', '')
    s+= _str('|T=',      PDE.t_full,               '%-7.3f', '')
    s+= _str('|R=',      PDE.u_calc_erank,         '%-5.1f', '')
    s+= _str('|It=',     PDE.algss_par['iters'],   '%-2d'  , '')
    return s
    
def compose_res(PDE):
    ''' Prepare string representation of the computation result.  '''
    ud_err = [None]*3
    if isinstance(PDE.ud_err, list):
        for i in range(len(PDE.ud_err)):
            ud_err[i] = PDE.ud_err[i]
    s = ''
    s+= _str('PDE                           : ', PDE.txt               , '%-s') 
    s+= _str('PDE mode                      : ', _mode2str(PDE.mode, 2), '%-s')  
    s+= _str('Used PDE solver               : ', PDE.solver_txt        , '%-s') 
    s+= _str('PDE dimension                 : ', PDE.dim               , '%-4d') 
    s+= _str('Value of d                    : ', PDE.d                 , '%-4d') 
    s+= _str('Mesh 1D size                  : ', PDE.n                 , '%-4d') 
    s+= _str('Solution erank                : ', PDE.u_calc_erank      , '%-6.2f') 
    s+= _str('Analit. sol. erank            : ', PDE.u_real_erank      , '%-6.2f') 
    s+= _str('Solution error                : ', PDE.u_err             , '%-8.2e') 
    s+= _str('X-derivative error            : ', ud_err[0]             , '%-8.2e') 
    s+= _str('Y-derivative error            : ', ud_err[1]             , '%-8.2e') 
    s+= _str('Z-derivative error            : ', ud_err[2]             , '%-8.2e') 
    s+= _str('Energy real                   : ', PDE.en_real           , '%-16.10f') 
    s+= _str('Energy calc                   : ', PDE.en_calc           , '%-16.10f') 
    s+= _str('Energy err                    : ', PDE.en_err            , '%-8.2e') 
    s+= _str('(u, f) real                   : ', PDE.uf_real           , '%-16.10f') 
    s+= _str('(u, f) calc                   : ', PDE.uf_calc           , '%-16.10f') 
    s+= _str('(u, f) err                    : ', PDE.uf_err            , '%-8.2e') 
    s+= _str('Solver iterations             : ', PDE.algss_par['iters'], '%-4d')    
    s+= _str('Matrix A erank                : ', PDE.a_erank           , '%-4d') 
    s+= _str('Vector rhs erank              : ', PDE.rhs_erank         , '%-4d') 
    s+= _str('Time: matrices generation (s.): ', PDE.t['mgen']         , '%-8.4f') 
    s+= _str('Time: system generation   (s.): ', PDE.t['sgen']         , '%-8.4f') 
    s+= _str('Time: system solution     (s.): ', PDE.t['solve']        , '%-8.4f') 
    s+= _str('Total time                (s.): ', PDE.t_full            , '%-8.4f') 
    s+= _str('*Time: prepare result     (s.): ', PDE.t['prepare_res']  , '%-8.4f') 
    return s

def compose_info(PDE):
    ''' Prepare string representation of the computation parameters.  '''
    s = '__________________General parameters\n'
    s+= _str('Mode             : ', _mode2str(PDE.mode, 2), '%-s')  
    s+= _str('Solver           : ', PDE.solver_txt        , '%-s')  
    s+= _str('Model num        : ', PDE.model_num         , '%-2d')  
    s+= _str('Parameters       : ', PDE._params2str()     , '%-s')
    s+= _str('with_en          : ', PDE.with_en           , '%-r') 
    s+= '__________________Verbosity parameters\n'
    s+= _str('verb_gen         : ', PDE.verb_gen          , '%-r') 
    s+= _str('verb_cross       : ', PDE.verb_cross        , '%-r') 
    s+= _str('verb_amen        : ', PDE.verb_amen         , '%-r') 
    s+= _str('print_to_std     : ', PDE.print_to_std      , '%-r')
    s+= _str('print_to_file    : ', PDE.print_to_file     , '%-r')
    s+= _str('out_file         : ', PDE.out_file          , '%-s')
    if PDE.mode != MODE_TT:
        return s
    s+= '__________________TT parameters\n'
    for name in PDE.tau.names:
        s+= '%-10s (round/cross/amens) : %-12s / %-12s / %-12s \n'%(name,\
             _str('', PDE.tau[name]['round'], '%-8.2e', '', 'None'),
             _str('', PDE.tau[name]['cross'], '%-8.2e', '', 'None'),
             _str('', PDE.tau[name]['amens'], '%-8.2e', '', 'None'))  
        s+= '           * %-s \n'%PDE.tau.txt[name]
    s+= _str('nswp_amen        : ', PDE.algss_par['nswp'] , '%-4d') 
    s+= _str('kickrank_amen    : ', PDE.algss_par['kickrank'], '%-4d')  
    s+= _str('Initial guess    : ', PDE.sol0 is not None  , '%-r') 
    return s
    
def _str(name, val, form='%-s', delim='\n', none_is=None):
    if val is None:
        if none_is == None:
            return ''
        return name + none_is + delim
    if val == '':
        return ''
    if isinstance(val, list) and len(val)==0:
        return '' 
    return name + form%val + delim

def _mode2str(mode, verb=0):
    if mode == MODE_NP:
        if verb==0:
            return 'np'
        if verb==1:
            return 'MODE_NP' 
        return 'MODE_NP (numpy format)'   
    if mode == MODE_TT:
        if verb==0:
            return 'tt'
        if verb==1:
            return 'MODE_TT' 
        return 'MODE_TT (tensor train format)'
    if mode == MODE_SP:
        if verb==0:
            return 'sp'
        if verb==1:
            return 'MODE_SP' 
        return 'MODE_SP (scipy sparse format)'
    return None