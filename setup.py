# -*- coding: utf-8 -*-
import os
from setuptools import setup

setup(
    name = 'qttpdesolver',
    version = '0.1',
    packages = ['qttpdesolver',
                'qttpdesolver.utils',
                'qttpdesolver.tensor_wrapper',
                'qttpdesolver.pde',
                'qttpdesolver.solvers', 
                'qttpdesolver.solvers.solver_fs', 
                'qttpdesolver.solvers.solver_fd', 
                'qttpdesolver.solvers.solver_fsx', 
                'qttpdesolver.solvers.solver_fdx', 
                'qttpdesolver.solvers.solver_fs_nh'],
    include_package_data = True,
    requires = ['python (>= 2.7)'],
    description  = 'Fast solver for partial differential equations based on low rank quantized tensor train decomposition, that can operate on huge grids.',
    long_description =  open(os.path.join(os.path.dirname(__file__), 'README.rst')).read(), 
    author = 'Andrei Chertkov and Ivan Oseledets',
    author_email = 'andrey.mipt@mail.ru ivan.oseledets@gmail.com',
    url = 'https://github.com/AndreChertkov/qttpdesolver',
    download_url = 'https://github.com/AndreChertkov/qttpdesolver',
    #license = 'BSD License',
    keywords = 'tensor train, diffusion equation, pde, solver, qtt-decomposition',
    classifiers = [
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Solvers',
        #'License :: OSI Approved :: BSD License',
    ],
)