#!/bin/bash

docker build -t amplab/ray:base docker/base
docker build -t amplab/ray:examples-base docker/examples-base
docker build -t amplab/ray:devel docker/devel
docker build -t amplab/ray:deploy docker/deploy
docker build -t amplab/ray:examples docker/examples
