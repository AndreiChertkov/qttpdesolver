qttpdesolver
============

1. Description
  Package qttpdesolver is a fast solver for partial differential equations,
  based on the new robust discretization scheme and on the low rank quantized
  tensor train decomposition (QTT-decomposition), that can operate on huge grids.

2. Requirements
  2.1. python 2.7;
  2.2. standard python packages like numpy, matplotlib, etc. (all of them are included in Anaconda distribution);
  2.3. ttpy python package (https://github.com/oseledets/ttpy).
  
3. Installation
  3.1. install python 2.7 and standard packages. The best way is to use Anaconda distribution from https://www.continuum.io/downloads;
  3.2. install ttpy python package according to instructions from https://github.com/oseledets/ttpy;
  3.3. download this repo and run "python setup.py install" from the root folder of the project.
  
4. Tests
  Run "python ./tests/test_all.py" from the root folder of the project to perform all tests (-v option is available).

5. Examples
  See "./examples/api_basic_*.ipynb" for API details.
  See over examples from "./examples/" folder for more details.
  All examples are performed as interactive browser-based jupyter notebooks (the corresponding package is included in Anaconda distribution).
  To work with example run "jupyter notebook" (it will be opened in the browser),
  find in the directory tree the "qttpdesolver/examples" folder,
  open the corresponding notebook and run all the cells one by one.
  
6. Authors
  Andrei V. Chertkov (andrey.mipt@mail.ru) and Ivan V. Oseledets (ivan.oseledets@gmail.com)

7. Related publications
  [1] I. V. Oseledets, M. V. Rakhuba, A. V. Chertkov, Black-box solver for multiscale modelling using the QTT format, in: Proc. ECCOMAS, Crete Island, Greece, 2016.
      URL https://www.eccomas2016.org/proceedings/pdf/10906.pdf