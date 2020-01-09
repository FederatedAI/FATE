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
module_name="roll"
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

packaging() {
    source ../../../default_configurations.sh
    package_init ${output_packages_dir} ${module_name}
    get_module_package ${source_code_dir} ${module_name} eggroll-${module_name}-${version}.tar.gz
    tar xzf eggroll-${module_name}-${version}.tar.gz
    rm -rf eggroll-${module_name}-${version}.tar.gz
}



config() {
    config_label=$4
	cd ${output_packages_dir}/config/${config_label}
    cd ./${module_name}/conf
	cp ${source_code_dir}/cluster-deploy/scripts/deploy/eggroll/services.sh ./
    sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./services.sh
    sed -i.bak "s#installdir=.*#installdir=${deploy_dir}#g" ./services.sh
    rm -rf ./services.sh.bak

    mkdir conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/roll.properties ./conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/log4j2.properties ./conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/applicationContext-roll.xml ./conf

    sed -i.bak "s/party.id=.*/party.id=${party_id}/g" ./conf/roll.properties
    sed -i.bak "s/service.port=.*/service.port=${port}/g" ./conf/roll.properties
    sed -i.bak "s/meta.service.ip=.*/meta.service.ip=${meta_service_ip}/g" ./conf/roll.properties
    sed -i.bak "s/meta.service.port=.*/meta.service.port=${meta_service_port}/g" ./conf/roll.properties

    sed -i.bak "s#property.logDir=.*#property.logDir=logs/${module_name}#g" ./conf/log4j2.properties
    rm -rf ./conf/log4j2.properties.bak ./conf/roll.properties.bak
}

init() {
    return 0
}

install(){
    mkdir -p ${deploy_dir}/
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
    cd ${deploy_dir}/${module_name}
    ln -s eggroll-${module_name}-${roll_version}.jar eggroll-${module_name}.jar
    mv ./services.sh ${deploy_dir}/
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
