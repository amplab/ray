# Installation on Ubuntu

Ray must currently be built from source. We have tested Ray on Ubuntu 14.04.

## Clone the Ray repository

```
git clone https://github.com/amplab/ray.git
```

## Dependencies

First install the dependencies. We currently do not support Python 3.

```
sudo apt-get update
sudo apt-get install -y git cmake build-essential autoconf curl libtool python-dev python-numpy python-pip libboost-all-dev unzip graphviz
pip install ipython funcsigs subprocess32 protobuf colorama graphviz cloudpickle
```

## Build

Then run the setup scripts.

```
cd ray
./setup.sh # Build all necessary third party libraries (e.g., gRPC and Apache Arrow). This may take about 10 minutes.
./build.sh # Build Ray.
```

Then to install the Python package, run the following.
```
cd lib/python
python setup.py develop --user
```
This adds a reference to the current directory, so you need to leave all the
files in place.

## Test if the installation succeeded

To test if the installation was successful, try running some tests.

```
python test/runtest.py # This tests basic functionality.
python test/array_test.py # This tests some array libraries.
```
