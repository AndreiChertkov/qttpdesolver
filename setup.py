# -*- coding: utf-8 -*-
import os
from setuptools import setup

def find_packages(package, basepath):
  packages = [package]
  for name in os.listdir(basepath):
    path = os.path.join(basepath, name)
    if not os.path.isdir(path):
      continue
    packages.extend(find_packages('%s.%s'%(package, name), path))
  return packages

setup(
    name = 'qttpdesolver',
    version = '0.1',
    packages = find_packages('qttpdesolver', './qttpdesolver/'),
    include_package_data = True,
    requires = ['python (>= 2.7)'],
    description  = 'Fast solver for partial differential equations based on low rank quantized tensor train decomposition, that can operate on huge grids.',
    long_description =  open(os.path.join(os.path.dirname(__file__), 'README.rst')).read(),
    author = 'Andrei Chertkov and Ivan Oseledets',
    author_email = 'andrey.mipt@mail.ru, ivan.oseledets@gmail.com',
    url = 'https://github.com/AndreChertkov/qttpdesolver',
    download_url = 'https://github.com/AndreChertkov/qttpdesolver',
    #license = 'BSD License',
    keywords = 'partial differential equation, pde solver, multiscale problem, diffusion equation, heat conduction, tensor train, qtt-decomposition',
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
