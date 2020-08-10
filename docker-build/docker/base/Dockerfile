FROM centos:centos7.2.1511

ARG version

ENV venv_dir /data/projects/python/venv
ENV miniconda3_dir /data/projects/python/miniconda3

COPY requirements.txt /data/projects/python/

RUN set -eux && \
    rpm --rebuilddb && \
    rpm --import /etc/pki/rpm-gpg/RPM* && \
    yum -y install gcc gcc-c++ make openssl-devel gmp-devel mpfr-devel libmpc-devel\
    libmpcdevel libaio numactl autoconf automake libtool libffi-devel  \
    snappy snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan lsof sysstat telnet psmisc && \
    yum clean all

RUN set -eux && \
    cd /data/projects/python/ && \
    curl -sL https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -o /data/projects/python/Miniconda3-4.5.4-Linux-x86_64.sh && \
    sh ./Miniconda3-*-Linux-x86_64.sh -b -p ${miniconda3_dir} && \
    rm -f ./Miniconda3-*-Linux-x86_64.sh
    
RUN set -eux && \
    cd /data/projects/python/  && \
    curl -L https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/python-env-${version}-release.tar.gz -o pip-packages-fate-${version}.tar.gz && \
    tar -xvf pip-packages-fate-${version}.tar.gz && \
    rm pip-packages-fate-${version}.tar.gz && \
    set -eux && \
    cd /data/projects/python/python-env  && \
    ls && \
    ${miniconda3_dir}/bin/pip install virtualenv-20.0.18-py2.py3-none-any.whl -f ./ --no-index && \ 
    ${miniconda3_dir}/bin/virtualenv -p ${miniconda3_dir}/bin/python3.6  --no-wheel --no-setuptools --no-download ${venv_dir} && \ 
    source ${venv_dir}/bin/activate  && \
    pip install ./setuptools-42.0.2-py2.py3-none-any.whl  && \
    pip install -r ../requirements.txt -f ./  && \
    cd ../ && \
    rm -rf ./python-env && \
    pip list | wc -l
