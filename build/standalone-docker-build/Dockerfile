#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

FROM centos/python-36-centos7

ARG version

USER root

WORKDIR /data/projects/fate
ENV WORKDIR /data/projects/fate

ADD fate.tar ${WORKDIR}
ADD docker-entrypoint.sh ${WORKDIR}

RUN rm /bin/sh && ln -sf /bin/bash /bin/sh

COPY docker-entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN sh ./bin/install_os_dependencies.sh root && yum clean all

# python enviroment and dependencies
RUN pip install --upgrade pip -f ./env/pypi --no-index
RUN pip install -r ./requirements.txt -f ./env/pypi --no-index
ENV PYTHONPATH ${WORKDIR}/fate/python:${WORKDIR}/fateflow/python
RUN sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=${PYTHONPATH}#g" ./bin/init_env.sh

# install and initialize the fate client
RUN cd ./fate/python/fate_client && python setup.py install
RUN flow init -c ./conf/service_conf.yaml

# java enviroment
RUN cd ./env/jdk/ && tar -xzf jdk-8u192.tar.gz
ENV JAVA_HOME ${WORKDIR}/env/jdk/jdk-8u192
RUN sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${JAVA_HOME}#g" ./bin/init_env.sh

RUN sed -i.bak "s#fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:${WORKDIR}/fate_sqlite.db#g" ./fateboard/conf/application.properties
RUN sed -i.bak "s#fateflow.url=.*#fateflow.url=http://localhost:9380#g" ./fateboard/conf/application.properties

ENV FATE_PROJECT_BASE ${WORKDIR}
ENV FATE_BASE ${WORKDIR}/fate
ENV FATE_FLOW_BASE ${WORKDIR}/fateflow

# clean up things that don't need to be used in the Docker environment
RUN sed -i.bak "s#venv=.*##g" ./bin/init_env.sh
RUN sed -i.bak "s#source.*venv.*##g" ./bin/init_env.sh
RUN rm -rf ./env/python36
RUN rm -rf ./init.sh

ENTRYPOINT ["docker-entrypoint.sh"]
