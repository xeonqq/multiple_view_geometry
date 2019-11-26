#!/bin/bash

which python
which python3
git clone https://github.com/uoip/g2opy.git /tmp/g2opy
cd /tmp/g2opy
mkdir build
cd build
cmake -DPYTHON_EXECUTABLE=$(which python) ..
make -j8
cd ..
python setup.py install


