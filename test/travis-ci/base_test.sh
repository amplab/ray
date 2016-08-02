#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
  docker run amplab/ray:test-base bash -c 'source setup-env.sh && cd test && python runtest.py && python array_test.py && python microbenchmarks.py'
else
  source setup-env.sh
  pushd test
    runtest.py
    python array_test.py
    python microbenchmarks.py
  popd test
fi
