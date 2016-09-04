# Installation on Mac OS X

Ray must currently be built from source. We have tested Ray on OS X 10.11.

## Clone the Ray repository

```
git clone https://github.com/amplab/ray.git
```

## Dependencies

First install the dependencies using brew. We currently do not support Python 3.
If you have trouble installing the Python packages, you may find it easier to
install [Anaconda](https://www.continuum.io/downloads).

```
brew update
brew install git cmake automake autoconf libtool boost graphviz
sudo easy_install pip
pip install ipython --user
pip install numpy funcsigs subprocess32 protobuf colorama graphviz cloudpickle --ignore-installed six
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
