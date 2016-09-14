# -*- coding: utf-8 -*-
import os
from setuptools import setup

setup(
    name = 'qttpdesolver',
    version = '0.1',
    packages = ['qttpdesolver', 'qttpdesolver.pde', 'qttpdesolver.pde.pde_utils',
                'qttpdesolver.solvers', 'qttpdesolver.utils'],
    include_package_data = True,
    requires = ['python (>= 2.7)'],
    description  = 'Fast solver for partial differential equations based on low rank quantized tensor train decomposition, that can operate on huge grids.',
    long_description =  open(os.path.join(os.path.dirname(__file__), 'README.rst')).read(), 
    author = 'Ivan Oseledets, Andrei Chertkov',
    author_email = 'ivan.oseledets@gmail.com, andrey.mipt@mail.ru',
    url = 'https://bitbucket.org/oseledets/1dqtt/',
    download_url = 'https://bitbucket.org/oseledets/1dqtt/master',
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