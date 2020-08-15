rm -rf build
python setup.py build_ext --compiler=msvc
python setup.py install --user
