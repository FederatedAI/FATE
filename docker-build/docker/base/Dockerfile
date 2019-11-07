FROM centos:centos7.2.1511

COPY pip-packages-fate-1.1.tar.gz /data/projects/python/
COPY requirements.txt /data/projects/python/

RUN yum -y install gcc gcc-c++ make openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-devel snappy snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan lsof &&\
    cd /data/projects/python/ && \
    curl https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh > /data/projects/python/Miniconda3-4.5.4-Linux-x86_64.sh && \
    miniconda3_dir=/data/projects/python/miniconda3 &&\
    venv_dir=/data/projects/python/venv && \
    python_version=1.1 && \
    sh ./Miniconda3-*-Linux-x86_64.sh -b -p ${miniconda3_dir} &&\
    tar -xvf pip-packages-fate-1.1.tar.gz && \
    rm pip-packages-fate-1.1.tar.gz && \
    ${miniconda3_dir}/bin/pip install virtualenv -f ./pip-packages-fate-${python_version} --no-index &&\
    ${miniconda3_dir}/bin/virtualenv -p ${miniconda3_dir}/bin/python3.6  --no-wheel --no-setuptools --no-download ${venv_dir}  &&\
    source ${venv_dir}/bin/activate &&\
    pip install ./pip-packages-fate-${python_version}/setuptools-41.4.0-py2.py3-none-any.whl &&\
    pip install -r ./requirements.txt -f ./pip-packages-fate-${python_version} --no-index