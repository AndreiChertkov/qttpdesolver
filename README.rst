qttpdesolver
============

1. Description
  Package qttpdesolver is a fast solver for partial differential equations,
  based on the new robust discretization scheme and on the low rank quantized
  tensor train decomposition (QTT-decomposition), that can operate on huge grids.

2. Requirements
  ttpy python package (https://github.com/oseledets/ttpy)

3. Installation
  python setup.py install
  
4. Examples
  See "./examples/api_basic_*.ipynb" for API details.
  See over examples from "./examples/" folder for more details.
  
5. Tests
  Type in console (from the root of the package) "python ./tests/test_all.py"
  to perform all tests (-v option is available).

6. Authors
  Ivan V. Oseledets  (ivan.oseledets@gmail.com)
  Andrei V. Chertkov (andrey.mipt@mail.ru)

7. Related publications
  [1] I. V. Oseledets, M. V. Rakhuba, A. V. Chertkov, Black-box solver for multiscale modelling using the QTT format, in: Proc. ECCOMAS, Crete Island, Greece, 2016.
      URL https://www.eccomas2016.org/proceedings/pdf/10906.pdf