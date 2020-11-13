FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3.8 python3-pip python3-setuptools
RUN apt-get update && apt-get install -y --no-install-recommends r-base
RUN apt-get update && apt-get install -y --no-install-recommends pandoc

MAINTAINER KC KC_Elephant_head

RUN pip3 install -U --trusted-host pypi.org --trusted-host files.pythonhosted.org pip
RUN pip3 install -U --trusted-host pypi.org --trusted-host files.pythonhosted.org plotly
RUN pip3 install -U --trusted-host pypi.org --trusted-host files.pythonhosted.org setuptools


RUN Rscript -e "install.packages('data.table', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN Rscript -e "install.packages('flexdashboard', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN Rscript -e "install.packages('rmarkdown', dependencies=TRUE, repos='http://cran.rstudio.com/')"

RUN mkdir /usr/src/TensorFlow_Experiments/
# COPY /TensorFlow_Experiments/ /usr/src/TensorFlow_Experiments/

# CMD["bash"]
# RUN cd /usr/src/TensorFlow_Experiments/
# RUN python3 setup.py clean --all install

