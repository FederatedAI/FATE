FROM centos:centos7.2.1511

RUN yum install -y \
       java-1.8.0-openjdk \
       java-1.8.0-openjdk-devel \
       bzip2

COPY cluster-deploy/scripts/fate-base /data/fate-base
COPY requirements.txt /data/fate-base/pip-dependencies/
RUN mkdir /data/fate-base/packages && mkdir /data/fate-base/pips

RUN curl https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh > /data/fate-base/packages/Miniconda3-4.5.4-Linux-x86_64.sh
RUN curl https://files.pythonhosted.org/packages/c2/d7/90f34cb0d83a6c5631cf71dfe64cc1054598c843a92b400e55675cc2ac37/pip-18.1-py2.py3-none-any.whl#sha256=7909d0a0932e88ea53a7014dfd14522ffef91a464daaaf5c573343852ef98550 > /data/fate-base/pips/pip-18.1-py2.py3-none-any.whl
RUN curl https://files.pythonhosted.org/packages/37/06/754589caf971b0d2d48f151c2586f62902d93dc908e2fd9b9b9f6aa3c9dd/setuptools-40.6.3-py2.py3-none-any.whl > /data/fate-base/pips/setuptools-40.6.3-py2.py3-none-any.whl
RUN curl https://files.pythonhosted.org/packages/7c/17/9b7b6cddfd255388b58c61e25b091047f6814183e1d63741c8df8dcd65a2/virtualenv-16.1.0-py2.py3-none-any.whl > /data/fate-base/pips/virtualenv-16.1.0-py2.py3-none-any.whl
RUN curl https://files.pythonhosted.org/packages/ff/47/1dfa4795e24fd6f93d5d58602dd716c3f101cfd5a77cd9acbe519b44a0a9/wheel-0.32.3-py2.py3-none-any.whl > /data/fate-base/pips/wheel-0.32.3-py2.py3-none-any.whl

RUN rpm --rebuilddb && sh /data/fate-base/env.sh
WORKDIR /data/fate-base/
RUN sed -ie "s#-f ./pip-dependencies --no-index##g" install_py3.sh
RUN sh /data/fate-base/install_py3.sh
