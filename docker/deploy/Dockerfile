# The deploy Docker image build a self-contained Ray instance suitable
# for end users.

FROM ubuntu:xenial
RUN apt-get update
RUN apt-get -y install apt-utils
RUN apt-get install -y git cmake build-essential autoconf curl libtool python-dev python-numpy python-pip libboost-all-dev unzip graphviz iputils-ping openssh-server
RUN pip install ipython typing funcsigs subprocess32 protobuf==3.0.0a2 colorama graphviz cloudpickle
WORKDIR /root
RUN git clone https://github.com/amplab/ray
WORKDIR /root/ray
RUN ./setup.sh
RUN ./build.sh
ENTRYPOINT /root/ray/setup-env.sh
