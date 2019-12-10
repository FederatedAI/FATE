# runtime envrionment
ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/python

COPY  fate_flow ./fate_flow
COPY  arch ./arch
COPY  federatedml ./federatedml
COPY  examples/ ./examples
COPY  eggroll-api-*.tar.gz /data/projects/fate/python

RUN mkdir -p eggroll/api && tar -C eggroll/api -xvf eggroll-api-*.tar.gz && \
    rm eggroll-api-*.tar.gz

ENV PYTHONPATH=/data/projects/fate/python
CMD sleep 5; source /data/projects/python/venv/bin/activate && cd ./fate_flow && \
    python fate_flow_server.py
