rm source/graspnetAPI.*
rm source/modules.rst
sphinx-apidoc -o ./source ../graspnetAPI
make clean
make html
make latex
cd build/latex
make
