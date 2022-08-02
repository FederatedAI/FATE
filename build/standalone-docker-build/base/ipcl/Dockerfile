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
ARG source_dir
ARG pip_index_url
ARG pip_index_host
ARG repo_file

USER root

WORKDIR /data/projects/fate
ENV WORKDIR /data/projects/fate

COPY requirements.txt install_os_dependencies.sh init.sh *.repo ${WORKDIR}/
RUN rm /bin/sh && ln -sf /bin/bash /bin/sh

RUN sh ./init.sh ${repo_file}

# fate python dependencies
RUN if [[ -z ${pip_index_url} ]]; then pip install --upgrade pip && pip install -r ./requirements.txt;\
    else pip_index_host=$(echo ${pip_index_url} | sed "s#http.*://##g" | sed "s#/.*##g") && \
    pip install --upgrade pip -i ${pip_index_url} --trusted-host ${pip_index_host} && \
    pip install -r ./requirements.txt -i ${pip_index_url} --trusted-host ${pip_index_host};fi
# clean
RUN rm -rf requirements.txt install_os_dependencies.sh init.sh *.repo

# upgrade gcc
RUN yum install -y git flex devtoolset-8 numactl-devel
SHELL ["/usr/bin/scl", "enable", "devtoolset-8"]

# install cmake
RUN wget http://cmake.org/files/v3.22/cmake-3.22.0.tar.gz \
    && tar zxf cmake-3.22.0.tar.gz && cd cmake-3.22.0 \
    && ./configure && make install -j \
    && cd .. && rm -rf cmake-*

# install nasm
RUN wget --no-check-certificate https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.gz \
    && tar zxf nasm-2.15.05.tar.gz && cd nasm-2.15.05 \
    && ./configure --prefix=/usr/local/nasm && make -j && make install \
    && ln -s /usr/local/nasm/bin/nasm /usr/bin/nasm \
    && ln -s /usr/local/nasm/bin/ndisasm /usr/bin/ndisasm \
    && cd .. && rm -rf nasm-*

# install openssl
RUN wget --no-check-certificate http://www.openssl.org/source/openssl-1.1.0l.tar.gz \
    && tar zxf openssl-1.1.0l.tar.gz && cd openssl-1.1.0l \
    && ./config --prefix=/usr/local/openssl && make -j && make install \
    && ln -sf /usr/local/openssl/bin/openssl /usr/bin/openssl \
    && echo "/usr/local/openssl/lib" >> /etc/ld.so.conf && ldconfig -v \
    && cd .. && rm -rf openssl-*
ENV OPENSSL_ROOT_DIR=/usr/local/openssl

# install ipcl-python
RUN git clone https://github.com/intel/pailliercryptolib_python.git \
    && cd pailliercryptolib_python && python setup.py install \
    && cd .. && rm -rf pailliercryptolib_python