# -*- coding: utf-8 -*- 
from ..tensor_wrapper import MODE_TT
  
def compose_model(PDE):
    ''' Prepare string representation of the PDE model.  '''
    s = ''
    s+= _str('PDE:        ', PDE.txt)
    s+= _str('Parameters: ', PDE._params2str())
    s+= _str('BC        : ', PDE.bc)
    s+= _str('', PDE.k_txt)
    s+= _str('', PDE.f_txt)
    s+= _str('', PDE.u_txt)
    s+= _str('', PDE.ux_txt)
    s+= _str('', PDE.uy_txt)
    s+= _str('', PDE.uz_txt)
    return s
        
def compose_res_1s(PDE):
    ''' Prepare one-string representation of the computation result.  '''
    s = ''
    s+= _str('d=',       PDE.d,                    '%2d'   , '')
    s+= _str('|',        PDE.solver_txt,           '%-2s'  , '')
    s+= _str('-',        PDE.mode,                 '%-2s'  , '')
    s+= _str('|',        PDE.bc,                   '%-2s'  , '')
    s+= _str('|er=',     PDE.u_err,                '%-8.1e', '')
    s+= _str('|erdx=',   PDE.ux_err,               '%-8.1e', '')
    s+= _str('|erdy=',   PDE.uy_err,               '%-8.1e', '')
    s+= _str('|erdz=',   PDE.uz_err,               '%-8.1e', '')
    s+= _str('|T=',      PDE.t_full,               '%8.3f' , '')
    s+= _str('|R=',      PDE.r['u_calc'],          '%6.1f' , '')
    if PDE.LSS.iters > 0:
        s+= _str('|It=',     PDE.LSS.iters,            '%2d'   , '')
    if PDE.u_err is None and PDE.mode == MODE_TT:
        s+= _str('|maxres=', PDE.LSS.max_res,          '%-8.1e', '')
    return s
    
def compose_res(PDE):
    ''' Prepare string representation of the computation result.  '''
    s = ''
    s+= _str('PDE                : ', PDE.txt               , '%-s') 
    s+= _str('PDE mode           : ', PDE.mode              , '%-s')  
    s+= _str('Used PDE solver    : ', PDE.solver_txt        , '%-s') 
    s+= _str('Boundary condition : ', PDE.bc                , '%-s') 
    s+= _str('PDE dimension      : ', PDE.dim               , '%-4d') 
    s+= _str('Value of d         : ', PDE.d                 , '%-4d') 
    s+= _str('Mesh 1D size       : ', PDE.n                 , '%-4d') 
    s+= _str('Solution erank     : ', PDE.r['u_calc']       , '%-6.2f') 
    s+= _str('Analit. sol. erank : ', PDE.r['u_real']       , '%-6.2f') 
    s+= _str('Solution error     : ', PDE.u_err             , '%-8.2e') 
    s+= _str('X-derivative error : ', PDE.ux_err            , '%-8.2e') 
    s+= _str('Y-derivative error : ', PDE.uy_err            , '%-8.2e') 
    s+= _str('Z-derivative error : ', PDE.uz_err            , '%-8.2e') 
    s+= _str('(u, f) real        : ', PDE.uf_real           , '%-16.10f') 
    s+= _str('(u, f) calc        : ', PDE.uf_calc           , '%-16.10f') 
    s+= _str('(u, f) err         : ', PDE.uf_err            , '%-8.2e') 
    s+= _str('Solver iterations  : ', PDE.LSS.iters         , '%-4d')    
    s+= _str('Matrix A erank     : ', PDE.r['A']            , '%-4d') 
    s+= _str('Vector rhs erank   : ', PDE.r['rhs']          , '%-4d') 
    s+= _str('Time: coeff.   (s.): ', PDE.t['cgen']         , '%-8.4f') 
    s+= _str('Time: matrices (s.): ', PDE.t['mgen']         , '%-8.4f') 
    s+= _str('Time: system   (s.): ', PDE.t['sgen']         , '%-8.4f') 
    s+= _str('Time: system   (s.): ', PDE.t['soln']         , '%-8.4f') 
    s+= _str('Total time     (s.): ', PDE.t_full            , '%-8.4f') 
    s+= _str('*Time: prep.   (s.): ', PDE.t['prep']         , '%-8.4f') 
    return s

def compose_info(PDE, full=False):
    ''' Prepare string representation of the computation parameters.  '''
    s = '__________________General parameters\n'
    s+= _str('Mode          : ', PDE.mode              , '%-s')  
    s+= _str('Solver        : ', PDE.solver_txt        , '%-s')  
    s+= _str('Boundary cond.: ', PDE.bc                , '%-s') 
    s+= _str('Model num     : ', PDE.model_num         , '%-2d')  
    s+= _str('Parameters    : ', PDE._params2str()     , '%-s')
    s+= '__________________Verbosity parameters\n'
    s+= _str('verb_gen      : ', PDE.verb_gen          , '%-r') 
    s+= _str('verb_crs      : ', PDE.verb_crs          , '%-r') 
    s+= _str('verb_lss      : ', PDE.verb_lss          , '%-r') 
    s+= _str('print_to_std  : ', PDE.print_to_std      , '%-r')
    s+= _str('print_to_file : ', PDE.print_to_file     , '%-r')
    s+= _str('out_file      : ', PDE.out_file          , '%-s')
    if not full and PDE.mode != MODE_TT:
        return s
    s+= '__________________TT parameters\n'
    s+= _str('tau                : ', PDE.tau               , '%-8.2e')
    s+= _str('eps_lss            : ', PDE.eps_lss           , '%-8.2e')
    s+= _str('tau_lss            : ', PDE.tau_lss           , '%-8.2e')
    s+= _str('tau_real           : ', PDE.tau_real          , '%-8.2e')
    s+= _str('lss: nswp          : ', PDE.LSS.nswp          , '%-4d') 
    s+= _str('lss: kickrank      : ', PDE.LSS.kickrank      , '%-4d')  
    s+= _str('lss: max full size : ', PDE.LSS.max_full_size , '%-4d') 
    s+= _str('lss: local restart : ', PDE.LSS.local_restart , '%-4d') 
    s+= _str('lss: local iters   : ', PDE.LSS.local_iters   , '%-4d') 
    s+= _str('lss: use sol0      : ', PDE.sol0 is not None  , '%-r') 
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