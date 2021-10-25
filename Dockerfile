From sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

RUN apt-get update &&\
    apt-get install -y build-essential \
    apt-transport-https \
    lsb-release  \
    ca-certificates \
    curl

RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs

RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install \
    ptvsd \
    jupyterlab>=2 \
    xeus-python \
    nb_black \
    ipdb \
    'streamlit==0.81.1' \
    stqdm

WORKDIR /project
RUN mkdir /project/neuraldistributions

COPY neuraldistributions /project/neuraldistributions
COPY setup.py /project

RUN python3 setup.py develop

RUN mkdir /project/lib
COPY lib /project/lib

RUN python -m pip install "git+https://github.com/mohammadbashiri/neuralpredictors.git@v0.1"
RUN python -m pip install nnfabrik==0.1.0
RUN python -m pip install -e /project/lib/nnsysident
RUN python -m pip install -e /project/lib/neuralmetrics

COPY ./jupyter/jupyter_notebook_config.py /root/.jupyter/