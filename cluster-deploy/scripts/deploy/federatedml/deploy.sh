#!/bin/bash

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
set -e
module_name="federatedml"
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./configurations.sh

usage() {
	echo "usage: $0 {binary/build} {packaging|config|install|init} {configurations path}."
}

deploy_mode=$1
config_path=$3
if [[ ${config_path} == "" ]] || [[ ! -f ${config_path} ]]
then
	usage
	exit
fi
source ${config_path}

# deploy functions

packaging() {
    source ../../default_configurations.sh
    package_init ${output_packages_dir} ${module_name}
	cp -r ${source_code_dir}/federatedml ${output_packages_dir}/source/${module_name}/
	cp -r ${source_code_dir}/examples ${output_packages_dir}/source/${module_name}
	cp -r ${source_code_dir}/federatedrec ${output_packages_dir}/source/${module_name}
	mkdir -p ${output_packages_dir}/source/${module_name}/arch
	cp -r ${source_code_dir}/arch/api ${output_packages_dir}/source/${module_name}/arch/
	return 0
}

config() {
    config_label=$4
	cd ${output_packages_dir}/config/${config_label}/${module_name}/conf
	python ${cwd}/generate_server_conf.py ${cwd}/service.env.tmp ./server_conf.json
	cp ${source_code_dir}/cluster-deploy/scripts/deploy/services.sh ./
	cp ${source_code_dir}/cluster-deploy/scripts/deploy/init_env.sh ./
	sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=${python_path}#g" ./init_env.sh
	sed -i.bak "s#venv=.*#venv=${venv_dir}#g" ./init_env.sh
    sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./init_env.sh
    rm -rf ./init_env.sh.bak
	return 0
}

install () {
    mkdir -p ${deploy_dir}
    cp -r ${deploy_packages_dir}/source/${module_name}/* ${deploy_dir}/

    # deal server conf
    mkdir -p ${deploy_dir}/arch/conf
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/arch/conf/

    # deal services.sh and init_env.sh
    fate_deploy_dir=`dirname ${deploy_dir}`
    mv ${deploy_dir}/arch/conf/services.sh ${fate_deploy_dir}/
    mv ${deploy_dir}/arch/conf/init_env.sh ${fate_deploy_dir}/
}

init (){
	return 0
}

case "$2" in
    packaging)
        packaging $*
        ;;
    config)
        config $*
        ;;
    install)
        install $*
        ;;
    init)
        init $*
        ;;
	*)
	    usage
        exit -1
esac
