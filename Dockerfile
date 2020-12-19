FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3.8 python3-pip python3-setuptools
RUN apt-get update && apt-get install -y --no-install-recommends r-base
RUN apt-get update && apt-get install -y --no-install-recommends pandoc
RUN apt-get update && apt-get install -y gfortran
RUN apt-get update && apt-get install -y libxml2
RUN apt-get update && apt-get install -y libxml2-dev
RUN apt-get update && apt-get install -y libtool

MAINTAINER KC KCreatiVision

RUN pip3 install -U --trusted-host pypi.org --trusted-host files.pythonhosted.org pip
RUN pip3 install -U --trusted-host pypi.org --trusted-host files.pythonhosted.org plotly
RUN pip3 install -U --trusted-host pypi.org --trusted-host files.pythonhosted.org setuptools

ADD . /usr/src/TensorFlow_Experiments/
WORKDIR /usr/src/TensorFlow_Experiments/

RUN ls -lh /usr/src/TensorFlow_Experiments/
RUN python3 /usr/src/TensorFlow_Experiments/setup.py clean --all install
