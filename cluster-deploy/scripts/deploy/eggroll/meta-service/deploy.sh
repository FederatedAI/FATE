#!/bin/bash

set -e
module_name="meta-service"
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
    if [[ "${deploy_mode}" == "binary" ]]; then
        get_module_binary ${source_code_dir} ${module_name} eggroll-${module_name}-${version}.tar.gz
        tar xzf eggroll-${module_name}-${version}.tar.gz
        rm -rf eggroll-${module_name}-${version}.tar.gz
    elif [[ "${deploy_mode}" == "build" ]]; then
        target_path=${source_code_dir}/eggroll/framework/${module_name}/target
        if [[ -f ${target_path}/eggroll-${module_name}-${version}.jar ]];then
            cp ${target_path}/eggroll-${module_name}-${version}.jar ${output_packages_dir}/source/${module_name}/
            cp -r ${target_path}/lib ${output_packages_dir}/source/${module_name}/
        else
            echo "[INFO] Build ${module_name} failed, ${target_path}/eggroll-${module_name}-${version}.jar: file doesn't exist."
        fi
    fi
}



config() {
    config_label=$4
	cd ${output_packages_dir}/config/${config_label}
    cd ./${module_name}/conf

	cp ${source_code_dir}/cluster-deploy/scripts/deploy/eggroll/services.sh ./
    sed -i "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./services.sh
    sed -i "s#installdir=.*#installdir=${deploy_dir}#g" ./services.sh

    mkdir conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/${module_name}.properties ./conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/log4j2.properties ./conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/applicationContext-${module_name}.xml ./conf

	sed -i "s/party.id=.*/party.id=${party_id}/g" ./conf/meta-service.properties
	sed -i "s/service.port=.*/service.port=${port}/g" ./conf/meta-service.properties
	sed -i "s#//.*?#//${db_ip}:3306/${db_name}?#g" ./conf/meta-service.properties
	sed -i "s/jdbc.username=.*/jdbc.username=${db_user}/g" ./conf/meta-service.properties
	sed -i "s/jdbc.password=.*/jdbc.password=${db_password}/g" ./conf/meta-service.properties
}

init() {
    return 0
}

install(){
    mkdir -p ${deploy_dir}/
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
    cd ${deploy_dir}/${module_name}
    ln -s eggroll-${module_name}-${version}.jar eggroll-${module_name}.jar
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