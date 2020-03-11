# runtime envrionment
ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/python

ARG version

COPY  fate_flow ./fate_flow
COPY  arch/api ./arch/api
COPY  arch/conf ./arch/conf
COPY  arch/core ./arch/core
COPY  arch/proto ./arch/proto
COPY  federatedml ./federatedml
COPY  federatedrec ./federatedrec
COPY  examples/ ./examples
COPY  eggroll-api-${version}.tar.gz ./

RUN mkdir -p eggroll/api && tar -C eggroll/api -xvf eggroll-api-${version}.tar.gz && \
    rm eggroll-api-${version}.tar.gz && \
    echo "source /data/projects/python/venv/bin/activate" >> ~/.bashrc

ENV PYTHONPATH=/data/projects/fate/python
CMD sleep 5; source /data/projects/python/venv/bin/activate && cd ./fate_flow && \
    python fate_flow_server.py
