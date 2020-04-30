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
module_name="fate_flow"
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
	  cp -r ${source_code_dir}/${module_name} ${output_packages_dir}/source
	return 0
}

config() {
    config_label=$4
    cd ${output_packages_dir}/config/${config_label}
    cp ${source_code_dir}/${module_name}/service.sh ./${module_name}/conf
    cp ${source_code_dir}/${module_name}/settings.py ./${module_name}/conf
    cd ./${module_name}/conf

    sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=${python_path}#g" ./service.sh
    sed -i.bak "s#venv=.*#venv=${venv_dir}#g" ./service.sh
    sed -i.bak "s/WORK_MODE =.*/WORK_MODE = 1/g" ./settings.py
    sed -i.bak "s/'user':.*/'user': '${db_user}',/g" ./settings.py
    sed -i.bak "s/'passwd':.*/'passwd': '${db_password}',/g" ./settings.py
    sed -i.bak "s/'host':.*/'host': '${db_ip}',/g" ./settings.py
    sed -i.bak "s/'port': 3306,/'port': ${db_port},/g" ./settings.py
    sed -i.bak "s/'name':.*/'name': '${db_name}',/g" ./settings.py
    sed -i.bak "s/'password':.*/'password': '${redis_password}',/g" ./settings.py
    sed -i.bak '$!N;s/REDIS = {\n.*'host'.*,/REDIS = \{\'$'\n    \'host\': "'${redis_ip}'",/' ./settings.py
    rm -rf ./service.sh.bak ./settings.py.bak
    return 0
}

install () {
    mkdir -p ${deploy_dir}/${module_name}
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
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
