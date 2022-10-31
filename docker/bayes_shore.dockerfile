# syntax=docker/dockerfile:1
FROM ubuntu:18.04
# create a directory to mount our data volume
RUN mkdir /project
#Install ubuntu libraires and packages
RUN apt-get update -y && \
    apt-get install git curl -y
#Set some environemnt variables we will need
ENV PATH="/build/miniconda3/bin:${PATH}"
ARG PATH="/build/miniconda3/bin:${PATH}"
RUN mkdir /build && \
    mkdir /build/.conda
#Install Python3.9 via miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /build/miniconda3 &&\
    rm -rf /Miniconda3-latest-Linux-x86_64.sh
WORKDIR /build
RUN conda install python=3.9
RUN conda install numpy seaborn scipy scikit-learn pandas
RUN conda install ipywidgets
RUN conda install -c conda-forge numpyro
RUN conda install -c conda-forge arviz
RUN mkdir /data
RUN pip install streamlit
EXPOSE 8501