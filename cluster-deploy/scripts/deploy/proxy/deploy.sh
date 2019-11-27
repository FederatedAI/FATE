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
module_name="proxy"
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
    source ../../default_configurations.sh
    package_init ${output_packages_dir} ${module_name}
    get_module_package ${source_code_dir} ${module_name} fate-${module_name}-${version}.tar.gz
    tar xzf fate-${module_name}-${version}.tar.gz
    rm -rf fate-${module_name}-${version}.tar.gz
}


config() {
    config_label=$4
    cd ${output_packages_dir}/config/${config_label}
    cd ./${module_name}/conf
	cp ${cwd}/service.sh ./
    sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./service.sh
    rm -rf ./service.sh.bak

    mkdir conf
    cp ${source_code_dir}/arch/networking/${module_name}/src/main/resources/applicationContext-${module_name}.xml ./conf
    cp ${source_code_dir}/arch/networking/${module_name}/src/main/resources/log4j2.properties ./conf
    cp ${source_code_dir}/arch/networking/${module_name}/src/main/resources/${module_name}.properties ./conf
    cp ${source_code_dir}/arch/networking/${module_name}/src/main/resources/route_tables/route_table.json ./conf
    sed -i.bak "s/port=.*/port=${proxy_port}/g" ./conf/${module_name}.properties
    sed -i.bak "s#route.table=.*#route.table=${deploy_dir}/${module_name}/conf/route_table.json#g" ./conf/${module_name}.properties
    sed -i.bak "s/coordinator=.*/coordinator=${party_id}/g" ./conf/${module_name}.properties
    sed -i.bak "s/ip=.*/ip=${proxy_ip}/g" ./conf/${module_name}.properties
    rm -rf ./conf/${module_name}.properties.bak

    cp ${cwd}/proxy_modify_json.py ./
    sed -i.bak "s/exchangeip=.*/exchangeip=\"${exchange_ip}\"/g" ./proxy_modify_json.py
    sed -i.bak "s/fip=.*/fip=\"${federation_ip}\"/g" ./proxy_modify_json.py
    sed -i.bak "s/flip=.*/flip=\"${fate_flow_ip}\"/g" ./proxy_modify_json.py
    sed -i.bak "s/sip1=.*/sip1=\"${serving_ip1}\"/g" ./proxy_modify_json.py
    sed -i.bak "s/sip2=.*/sip2=\"${serving_ip2}\"/g" ./proxy_modify_json.py
    sed -i.bak "s/partyId=.*/partyId=\"${party_id}\"/g" ./proxy_modify_json.py
    python proxy_modify_json.py ${module_name} ./conf/route_table.json
    rm -rf ./proxy_modify_json.py.bak
}

init (){
    return 0
}

install(){
    mkdir -p ${deploy_dir}/
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
    cd ${deploy_dir}/${module_name}
    ln -s fate-${module_name}-${proxy_version}.jar fate-${module_name}.jar
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
