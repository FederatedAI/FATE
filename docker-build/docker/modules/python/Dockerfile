# runtime envrionment
ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/python

ARG version

COPY  python /data/projects/fate/python
COPY  eggroll /data/projects/fate/eggroll
COPY  examples /data/projects/fate/examples
COPY  fate.env /data/projects/fate/

RUN echo "source /data/projects/python/venv/bin/activate" >> ~/.bashrc && \
    mkdir -p ./fml_agent/data

ENV PYTHONPATH=/data/projects/fate/python:/data/projects/fate/eggroll/python
ENV EGGROLL_HOME=/data/projects/fate/eggroll

CMD sleep 5; source /data/projects/python/venv/bin/activate && python ./fate_flow/fate_flow_server.py
