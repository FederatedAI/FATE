#!/bin/bash

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
    if [[ "${deploy_mode}" == "binary" ]]; then
        get_module_binary ${source_code_dir} ${module_name} ${module_name}-${version}.jar
    elif [[ "${deploy_mode}" == "build" ]]; then
        if [[ -f "${source_code_dir}/${module_name}/target/${module_name}-${version}.jar" ]];then
            cp ${source_code_dir}/${module_name}/target/${module_name}-${version}.jar ./
        else
            echo "[INFO] Build ${module_name} failed, ${source_code_dir}/${module_name}/target/${module_name}-${version}.jar: file doesn't exist."
        fi
    fi
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
    ln -s ${module_name}-${version}.jar ${module_name}.jar
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