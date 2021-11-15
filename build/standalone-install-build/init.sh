#!/usr/bin/env bash

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

basepath=$(cd `dirname $0`;pwd)
version=
env_dir=${basepath}/env

python_resouce=${basepath}/env/python36
pypi_resource=${basepath}/env/pypi
jdk_resource=${basepath}/env/jdk

jdk_dir=${jdk_resource}/jdk-8u192
miniconda_dir=${python_resouce}/miniconda
venv_dir=${python_resouce}/venv

echo "[INFO] env dir: ${env_dir}"
echo "[INFO] jdk dir: ${jdk_dir}"
echo "[INFO] venv dir: ${venv_dir}"

init() {

  cd ${basepath}

  echo "[INFO] install os dependency"
  sh bin/install_os_dependencies.sh
  echo "[INFO] install os dependency done"

  echo "[INFO] install python36"
  if [ ! -d ${miniconda_dir} ]; then
    bash ${python_resouce}/Miniconda3-4.5.4-Linux-x86_64.sh -b -p ${miniconda_dir}
  fi
  echo "[INFO] install python36 done"

  echo "[INFO] install jdk"
  if [ ! -d ${jdk_dir} ]; then
    cd ${jdk_resource}
    tar xzf jdk-8u192.tar.gz
  fi
  echo "[INFO] install jdk done"

  cd ${basepath}

  if [ ! -f ${venv_dir}/bin/python ]; then
    echo "[INFO] install virtualenv"
    ${miniconda_dir}/bin/pip install virtualenv -f ${pypi_resource} --no-index
    ${miniconda_dir}/bin/virtualenv -p ${miniconda_dir}/bin/python3.6  --no-wheel --no-setuptools --no-download ${venv_dir}
    source ${venv_dir}/bin/activate
    pip install setuptools --no-index -f ${pypi_resource}
    echo "[INFO] install virtualenv done"

    echo "[INFO] install python dependency packages by ${basepath}/requirements.txt using ${pypi_resource}"
    pip install -r ${basepath}/requirements.txt -f ${pypi_resource} --no-index
    echo "[INFO] install python dependency packages done"

    echo "[INFO] install fate client"
    cd ${basepath}/fate/python/fate_client
    python setup.py install
    flow init -c ${basepath}/conf/service_conf.yaml
    echo "[INFO] install fate client done"
  fi

  echo "[INFO] setup fateflow"
  sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=${basepath}/fate/python:${basepath}/fateflow/python#g" ${basepath}/bin/init_env.sh
  sed -i.bak "s#venv=.*#venv=${venv_dir}#g" ${basepath}/bin/init_env.sh
  sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${jdk_dir}/#g" ${basepath}/bin/init_env.sh
  echo "[INFO] setup fateflow done"
	#sed -i.bak "s#host:.*#host: 127.0.0.1#g" ${basepath}/conf/service_conf.yaml
  echo "[INFO] setup fateboard"
  sed -i.bak "s#fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:${basepath}/fate_sqlite.db#g" ${basepath}/fateboard/conf/application.properties
  sed -i.bak "s#fateflow.url=.*#fateflow.url=http://localhost:9380#g" ${basepath}/fateboard/conf/application.properties
  echo "[INFO] setup fateboard done"
}

action() {
  cd $basepath

  source $basepath/bin/init_env.sh

	cd  $basepath/fateflow
	sh  bin/service.sh $1

	cd  $basepath/fateboard
	sh  service.sh $1

	cd $basepath
}


case "$1" in
    start)
        action $@
        ;;
		
    stop)
        action $@
        ;;
		
    status)
        action $@
        ;;

    init)
	init
	;;
    *)
        echo "usage: $0 {start|stop|status|init}"
        exit -1
esac
