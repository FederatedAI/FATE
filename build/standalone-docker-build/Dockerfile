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

WORKDIR /fate
ENV WORKDIR /fate

ADD fate.tar /fate
ADD docker-entrypoint.sh /fate

RUN rm /bin/sh && ln -sf /bin/bash /bin/sh

COPY docker-entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN sh ./bin/install_os_dependencies.sh root && yum clean all

RUN pip install --upgrade pip

RUN pip install -r ./requirements.txt -f ./env/pypi --no-index

RUN cd ./fate/python/fate_client && python setup.py install
RUN flow init -c ./conf/service_conf.yaml

ENV PYTHONPATH /fate/fate/python:/fate/fateflow/python

RUN cd ./env/jdk/ && tar -xzf jdk-8u192.tar.gz
ENV JAVA_PATH /fate/env/jdk/jdk-8u192/bin

RUN sed -i.bak "s#fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:/fate/fate_sqlite.db#g" ./fateboard/conf/application.properties
RUN sed -i.bak "s#fateflow.url=.*#fateflow.url=http://localhost:9380#g" ./fateboard/conf/application.properties

ENV FATE_PROJECT_BASE /fate
ENV FATE_BASE /fate/fate
ENV FATE_FLOW_BASE /fate/fateflow

ENTRYPOINT ["docker-entrypoint.sh"]
