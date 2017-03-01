# -*- coding: utf-8 -*-
from IPython.display import display, Latex

class Table(object):
    
    def __init__(self, hrow, rows, fmts, lbl='', cpt='',
                 bold_hrow=True, bold_hcol=True, pos='t', elem2tex=True):
        if not isinstance(fmts[0], list):
            fmts = [fmts] * len(rows)
        self.tbl = '\\begin{figure}[%s]\n\t\\begin{center}\n'%pos
        self.tbl+= '\t\t\\captionof{table}{%s}\n'%cpt
        self.tbl+= '\t\t\\label{%s}\n\t\t\\begin{tabular}{ '%lbl
        self.tbl+= '| l '*len(hrow) + '|}\n\t\t\t\\hline\n\t\t\t'
        for elem in hrow:
            elem = elem.replace(' ', '\\,')
            if elem2tex: elem = self._elem2tex(elem)
            if bold_hrow: self.tbl+= '\\bf{%-s} & '%elem
            else: self.tbl+= '%-s & '%elem
        self.tbl = self.tbl[:-2] + '\n\t\t\\\\ \\hline '
        for i, row in enumerate(rows):
            if i>0:
                self.tbl+=' \n\t\t\t\\\\ '
            for j, elem in enumerate(row):
                if 's' in fmts[i][j]:
                    elem = elem.replace(' ', '\\,')
                if j==0 and bold_hcol: self.tbl+= ' \\bf{%-s} & '%(fmts[i][j]%elem)
                else: self.tbl+= ' %-s & '%(fmts[i][j]%elem)
            self.tbl = self.tbl[:-2]
        self.tbl+= ' \n\t\t\t\\\\ \\hline\n\t\t\\end{tabular}\n '
        self.tbl+= ' \t\\end{center}\n\\end{figure} '

    def _elem2tex(self, elem):
        self.vector_mask = '\\qttvect{%s}'
        self.matrix_mask = '\\qttmatr{%s}'
        if elem == 'd':
            return '$%s$'%elem
        if '_calc' in elem:
            elem = elem.replace('_calc', '')
        if '_real' in elem:
            elem = elem.replace('_real', '')
        if 'x' == elem[-1]:
            elem = elem.replace('x', '_x')
        if 'y' == elem[-1]:
            elem = elem.replace('y', '_y')
        if 'z' == elem[-1]:
            elem = elem.replace('z', '_z')
        if elem[0] == 'i':
            elem = elem[1:] + '^{-1}'
        if elem[0].lower() == elem[0]:
            return '$%s$'%(self.vector_mask%elem)
        else:
            return '$%s$'%(self.matrix_mask%elem)

    def present(self, rem, verb=True):
        if not verb: return
        tbl = self.tbl.split('\\begin{tabular}')[1].split('\\end{tabular}')[0]
        tbl = '\\begin{array}' + tbl + '\\end{array}'
        for symb in rem: tbl = tbl.replace(symb, '')
        display(Latex(tbl))
        
    def save(self, fname=None):
        if not fname: return
        with open(fname, 'w') as f:
            f.writelines(self.tbl.replace('\\,', ' '))