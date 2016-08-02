#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
  sudo apt-get update
  sudo apt-get -y install apt-transport-https ca-certificates
  sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
  echo deb https://apt.dockerproject.org/repo ubuntu-trusty main | sudo tee --append /etc/apt/sources.list.d/docker.list
  sudo apt-get update
  sudo apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install docker-engine
  docker version
  tar --exclude './docker' -cv . > ./docker/test-base/ray.tar
  docker build --shm-size=500m --no-cache -t amplab/ray:test-base docker/test-base
  rm ./docker/test-base/ray.tar
  docker build --no-cache -t amplab/ray:test-examples docker/test-examples
  docker ps -a
else
  ./install-dependencies.sh
  ./setup.sh
  ./build.sh
fi
