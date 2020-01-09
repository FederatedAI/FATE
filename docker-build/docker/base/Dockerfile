FROM centos:centos7.2.1511

ARG python_version

ENV venv_dir /data/projects/python/venv
ENV miniconda3_dir /data/projects/python/miniconda3

COPY requirements.txt /data/projects/python/

RUN set -eux; \
    rpm --rebuilddb; \
    rpm --import /etc/pki/rpm-gpg/RPM*; \
    \
    yum -y install gcc gcc-c++ make openssl-devel supervisor \
    gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf \
    automake libtool libffi-devel snappy snappy-devel zlib zlib-devel \
    bzip2 bzip2-devel lz4-devel libasan lsof \
    ;\
    yum clean all;

RUN set -eux; \
    cd /data/projects/python/ ; \
    curl https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh > /data/projects/python/Miniconda3-4.5.4-Linux-x86_64.sh ; \
    \
    sh ./Miniconda3-*-Linux-x86_64.sh -b -p ${miniconda3_dir} ; \
    rm -f ./Miniconda3-*-Linux-x86_64.sh;

RUN set -eux; \
    cd /data/projects/python/ ; \
    curl https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/pip-packages-fate-${python_version}.tar.gz > ./pip-packages-fate-${python_version}.tar.gz; \
    tar -xvf pip-packages-fate-${python_version}.tar.gz; \
    rm pip-packages-fate-${python_version}.tar.gz ; \
    \
    ${miniconda3_dir}/bin/pip install virtualenv -f ./pip-packages-fate-${python_version} --no-index ;\
    ${miniconda3_dir}/bin/virtualenv -p ${miniconda3_dir}/bin/python3.6  --no-wheel --no-setuptools --no-download ${venv_dir}  ;\
    \
    source ${venv_dir}/bin/activate ;\
    pip install ./pip-packages-fate-${python_version}/setuptools-41.4.0-py2.py3-none-any.whl ;\
    pip install -r ./requirements.txt -f ./pip-packages-fate-${python_version} ; \
    rm -rf ./pip-packages-fate-${python_version}; \
    python -V;
