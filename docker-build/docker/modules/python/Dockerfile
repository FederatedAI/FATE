# runtime envrionment
ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

RUN mkdir -p /data/projects/fate
COPY  fate_flow /data/projects/fate/python/fate_flow
COPY  arch /data/projects/fate/python/arch
COPY  federatedml /data/projects/fate/python/federatedml
COPY  eggroll-api-1.1.tar.gz /data/projects/fate/python

RUN cd /data/projects/fate/python && \
    mkdir -p eggroll/api && tar -C eggroll/api -xvf eggroll-api-1.1.tar.gz && \
    rm eggroll-api-1.1.tar.gz

WORKDIR /data/projects/fate/python

ENV PYTHONPATH=/data/projects/fate/python
CMD sleep 5; source /data/projects/python/venv/bin/activate && cd /data/projects/fate/python/fate_flow && \
    python fate_flow_server.py