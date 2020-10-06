#! /bin/bash
pip install .

cd graspnms
pip install .

cd ../meshpy
python setup.py develop

cd ../dex-net
python setup.py develop