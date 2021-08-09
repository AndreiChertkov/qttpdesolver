qttpdesolver
============

* **Description**

  Package qttpdesolver is a fast solver for partial differential equations,
  based on the new robust discretization scheme and on the low rank quantized
  tensor train (QTT) decomposition, that can operate on huge grids.

* **Requirements**

2.1. python 2.7;

2.2. standard python packages like numpy, matplotlib, etc. (all of them are included in Anaconda distribution);

2.3. ttpy python package (https://github.com/oseledets/ttpy).

* **Installation**

3.1. install python 2.7 and standard packages. The best way is to use Anaconda distribution from https://www.continuum.io/downloads;

3.2. install ttpy python package according to instructions from https://github.com/oseledets/ttpy;

3.3. download this repo and run "python setup.py install" from the root folder of the project.

* **Tests**
  Run

    ```bash
        python ./tests/test_all.py
    ```
from the root folder of the project to perform all tests (-v option is available).

* **Examples**

  All examples are performed as interactive browser-based jupyter notebooks (the corresponding package is included in Anaconda distribution).

  To work with example run "jupyter notebook" (it will be opened in the browser),
  find in the directory tree the "qttpdesolver/examples" folder,
  open the corresponding notebook and run all the cells one by one.
  See "./examples/api_basic_*.ipynb" for API details and over examples from "./examples/" folder for more details.

* **Authors**

  * Andrei Chertkov (andrey.mipt@mail.ru)
  * Ivan Oseledets (ivan.oseledets@gmail.com)

* **Related publications**

    1. [Robust discretization in quantized tensor train format for elliptic problems in two dimensions](https://arxiv.org/pdf/1612.01166.pdf)

        > A. V. Chertkov, I. V. Oseledets, M. V. Rakhuba

    1. [Black-box solver for multiscale modelling using the QTT format](https://www.eccomas2016.org/proceedings/pdf/10906.pdf)

        > I. V. Oseledets, M. V. Rakhuba, A. V. Chertkov
