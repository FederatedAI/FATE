# runtime envrionment
ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/python

COPY  fate_flow ./fate_flow
COPY  arch/api ./arch/api
COPY  arch/conf ./arch/conf
COPY  examples/ ./examples
COPY  fml_manager/ ./

RUN source /data/projects/python/venv/bin/activate && \
    pip uninstall -y ipykernel && \
    pip install notebook

ENV PYTHONPATH=/data/projects/fate/python
CMD sleep 5; source /data/projects/python/venv/bin/activate && jupyter notebook --ip=0.0.0.0 --port=20000 --allow-root --debug --no-browser --NotebookApp.token='' --NotebookApp.password=''
