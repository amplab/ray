FROM vfang/ray:deploy
RUN apt-get -y install vim
WORKDIR /bin
Copy run .
Copy scripts/ /root/ray/scripts
WORKDIR /root/ray
ENV PYTHONPATH="/root/ray/lib/python/:/root/ray/thirdparty/numbuf/build"
ENTRYPOINT []
