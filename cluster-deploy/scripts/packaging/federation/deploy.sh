#!/bin/bash

set -e
module_name="federation"
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
    if [[ "${deploy_mode}" == "binary" ]]; then
        get_module_binary ${source_code_dir} ${module_name} fate-${module_name}-${version}.tar.gz
        tar xzf fate-${module_name}-${version}.tar.gz
        rm -rf fate-${module_name}-${version}.tar.gz
    elif [[ "${deploy_mode}" == "build" ]]; then
        target_path=${source_code_dir}/arch/driver/${module_name}/target
        if [[ -f ${target_path}/fate-${module_name}-${version}.jar ]];then
            cp ${target_path}/fate-${module_name}-${version}.jar ${output_packages_dir}/source/${module_name}/
            cp -r ${target_path}/lib ${output_packages_dir}/source/${module_name}/
        else
            echo "[INFO] Build ${module_name} failed, ${target_path}/fate-${module_name}-${version}.jar: file doesn't exist."
        fi
    fi
}



config() {
    node_label=$4
    cd ${output_packages_dir}/config/${node_label}

    cd ./${module_name}/conf
	cp ${cwd}/service.sh ./
    sed -i "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./service.sh

    mkdir conf
    cp  ${source_code_dir}/arch/driver/${module_name}/src/main/resources/federation.properties ./conf
    cp  ${source_code_dir}/arch/driver/${module_name}/src/main/resources/log4j2.properties ./conf
    cp  ${source_code_dir}/arch/driver/${module_name}/src/main/resources/applicationContext-federation.xml ./conf

    sed -i "s/party.id=.*/party.id=${party_id}/g" ./conf/federation.properties
    sed -i "s/service.port=.*/service.port=${port}/g" ./conf/federation.properties
    sed -i "s/meta.service.ip=.*/meta.service.ip=${meta_service_ip}/g" ./conf/federation.properties
    sed -i "s/meta.service.port=.*/meta.service.port=${meta_service_port}/g" ./conf/federation.properties
}

init() {
    return 0
}

install(){
    mkdir -p ${deploy_dir}/
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
    cd ${deploy_dir}/${module_name}
    ln -s fate-${module_name}-${version}.jar fate-${module_name}.jar
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