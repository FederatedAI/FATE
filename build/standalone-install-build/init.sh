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
version=1.6.1
env_dir=${basepath}/env
python_dir=${basepath}/env/python36
pypi_dir=${basepath}/env/pypi
jdk_dir=${basepath}/env/jdk
miniconda_dir=${python_dir}/miniconda
venv_dir=${python_dir}/venv

init() {

  cd ${basepath}
  #install os dependency
  sh bin/init_env.sh

  #install python36
  if [ ! -f ${basepath}/miniconda/bin/python ]; then
    bash ${env_dir}/python36/Miniconda3-4.5.4-Linux-x86_64.sh -b -p ${miniconda_dir}
  fi

  if [ ! -f ${venv_dir}/bin/python ]; then
    #install python36
    ${miniconda_dir}/bin/pip install virtualenv -f ${pypi_dir}--no-index
    ${miniconda_dir}/bin/virtualenv -p ${miniconda_dir}/bin/python3.6  --no-wheel --no-setuptools --no-download ${venv_dir}
    source ${venv_dir}/bin/activate
    pip install ${pypi_dir}/setuptools-42.0.2-py2.py3-none-any.whl
    #install fate python dependency package
    echo "pip install -r ${basepath}/requirements.txt -f ${pypi_dir} --no-index"
    pip install -r ${basepath}/requirements.txt -f ${pypi_dir} --no-index
    pnum=$( pip list | wc -l )
    rnum=$( grep -cE '=|>|<' ${basepath}/requirements.txt  )
    echo "install: $pnum require: $rnum"

    #if [ $pnum -lt $rnum ]
    #then
    #  pip install -r ${basepath}/files/requirements.txt -f ${basepath}/files/pip-packages-fate-${version} --no-index
    #fi
    #rm -rf  ${basepath}/files
  fi

	#set fate_flow 
	sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=${basepath}/fate/python:${basepath}/fateflow/python#g" ${basepath}/bin/init_env.sh
	sed -i.bak "s#venv=.*#venv=${venv_dir}#g" ${basepath}/bin/init_env.sh
	sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${jdk_dir}/jdk1.8.0_192/#g" ${basepath}/bin/init_env.sh
	
	#sed -i.bak "s#host:.*#host: 127.0.0.1#g" ${basepath}/conf/service_conf.yaml
	
	#set board
	sed -i.bak "s#fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:${basepath}/fate_sqlite.db#g" ${basepath}/fateboard/conf/application.properties
	sed -i.bak "s#fateflow.url=.*#fateflow.url=http://localhost:9380#g" ${basepath}/fateboard/conf/application.properties

  action restart
}

action() {
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