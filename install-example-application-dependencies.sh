#!/usr/bin/env bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

platform="unknown"
unamestr="$(uname)"
if [[ "$unamestr" == "Linux" ]]; then
  echo "Platform is linux."
  platform="linux"
elif [[ "$unamestr" == "Darwin" ]]; then
  echo "Platform is macosx."
  platform="macosx"
else
  echo "Unrecognized platform."
  exit 1
fi

if [[ $platform == "linux" ]]; then
  # install scipy
  sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
  sudo pip install scipy
  # install tensorflow
  export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
  sudo pip install --upgrade $TF_BINARY_URL
  # install gym
  sudo pip install gym
elif [[ $platform == "macosx" ]]; then
  # install scipy
  sudo pip install scipy --ignore-installed six
  # install tensorflow
  export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl
  sudo pip install --upgrade $TF_BINARY_URL --ignore-installed six
  # install gym
  sudo pip install gym --ignore-installed six
fi
