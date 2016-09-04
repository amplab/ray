#!/bin/bash

set -x
set -e

TP_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
PREFIX=$TP_DIR/installed

# Determine how many parallel jobs to use for make based on the number of cores
unamestr="$(uname)"
if [[ "$unamestr" == "Linux" ]]; then
  PARALLEL=$(nproc)
elif [[ "$unamestr" == "Darwin" ]]; then
  PARALLEL=$(sysctl -n hw.ncpu)
  echo "Platform is macosx."
else
  echo "Unrecognized platform."
  exit 1
fi

echo "building numbuf"
cd $TP_DIR/numbuf
./setup.sh
./build.sh
python setup.py develop --user

echo "building GRPC"
cd $TP_DIR/grpc
make static HAS_SYSTEM_PROTOBUF=false HAS_SYSTEM_ZLIB=false HAS_SYSTEM_OPENSSL_ALPN=false HAS_SYSTEM_OPENSSL_NPN=false -j$PARALLEL

echo "building hiredis"
cd $TP_DIR/hiredis
make
