# runtime envrionment
ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

COPY . /data/projects/fate/python

WORKDIR /data/projects/fate

ENV PYTHONPATH=/data/projects/fate/python
CMD while true; do sleep 1; done;

