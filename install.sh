#! /bin/bash
python -m pip install .

cd graspnms
python -m pip install .

cd ../meshpy
python setup.py develop

cd ../dex-net
python setup.py develop
