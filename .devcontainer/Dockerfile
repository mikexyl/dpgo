FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install build-essential cmake-gui libsuitesparse-dev libboost-all-dev libeigen3-dev libgoogle-glog-dev \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y install software-properties-common && \
  rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:borglab/gtsam-release-4.2
RUN apt update && apt install -y libgtsam-dev libgtsam-unstable-dev && \
  rm -rf /var/lib/apt/lists/*