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

FROM python:3.6

WORKDIR /fate

ADD fate.tar /fate
ADD docker-entrypoint.sh /fate

RUN rm /bin/sh && ln -sf /bin/bash /bin/sh

COPY docker-entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN apt-get update && apt-get install libgmp3-dev -y && apt-get install -y libmpfr-dev libmpfr-doc libmpfr6 && apt-get install libmpc-dev -y

#RUN pip install -r requirements.txt -f ./pip-dependedocker ncies --no-index
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /fate/python/requirements.txt

ENV PYTHONPATH /fate/python

RUN tar -xzf /fate/jdk-8u192.tar.gz
ENV JAVA_PATH /fate/jdk-8u192/bin

ENTRYPOINT ["docker-entrypoint.sh"]