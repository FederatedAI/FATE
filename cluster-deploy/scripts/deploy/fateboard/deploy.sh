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
module_name="fateboard"
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
    get_module_package ${source_code_dir} ${module_name} ${module_name}-${fateboard_version}.jar
}

config() {
    config_label=$4
    cd ${output_packages_dir}/config/${config_label}
    cd ./${module_name}/conf
	cp ${cwd}/service.sh ./

    mkdir conf ssh
    touch ./ssh/ssh.properties

    cp ${source_code_dir}/${module_name}/src/main/resources/application.properties ./conf
    sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./service.sh
    sed -i.bak "s#^server.port=.*#server.port=${fateboard_port}#g" ./conf/application.properties
    sed -i.bak "s#^fateflow.url=.*#fateflow.url=http://${fate_flow_ip}:${fate_flow_port}#g" ./conf/application.properties
    sed -i.bak "s#^spring.datasource.driver-Class-Name=.*#spring.datasource.driver-Class-Name=com.mysql.cj.jdbc.Driver#g" ./conf/application.properties
    sed -i.bak "s#^spring.datasource.url=.*#spring.datasource.url=jdbc:mysql://${db_ip}:3306/${db_name}?characterEncoding=utf8\&characterSetResults=utf8\&autoReconnect=true\&failOverReadOnly=false\&serverTimezone=GMT%2B8#g" ./conf/application.properties
    sed -i.bak "s/^spring.datasource.username=.*/spring.datasource.username=${db_user}/g" ./conf/application.properties
    sed -i.bak "s/^spring.datasource.password=.*/spring.datasource.password=${db_password}/g" ./conf/application.properties
    rm -rf ./conf/application.properties.bak
    for node in "${node_list[@]}"
    do
        echo ${node}
        node_info=(${node})
        sed -i.bak "/${node_info[0]}/d" ./ssh/ssh.properties
        echo "${node_info[0]}=${node_info[1]}|${node_info[2]}|${node_info[3]}" >> ./ssh/ssh.properties
    done
    rm -rf ./ssh/ssh.properties.bak
}


install() {
    mkdir -p ${deploy_dir}/
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
    cd ${deploy_dir}/${module_name}
    ln -s ${module_name}-${fateboard_version}.jar ${module_name}.jar
}

init() {
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
