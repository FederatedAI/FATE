#!/bin/bash

set -e
module_name="egg"
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
        mkdir ./egg-manager
        cd ./egg-manager
        get_module_binary ${source_code_dir} ${module_name} eggroll-${module_name}-${version}.tar.gz
        tar xzf eggroll-${module_name}-${version}.tar.gz
        rm -rf eggroll-${module_name}-${version}.tar.gz
        cd ../

        mkdir ./egg-services
        cd ./egg-services

        mkdir storage-service-cxx
        cd storage-service-cxx
        get_module_binary ${source_code_dir} ${module_name} eggroll-storage-service-cxx-${version}.tar.gz
        tar xzf eggroll-storage-service-cxx-${version}.tar.gz
        rm -rf eggroll-storage-service-cxx-${version}.tar.gz
        get_module_binary ${source_code_dir} ${module_name} third_party_eggrollv1.tar.gz
        tar xzf third_party_eggrollv1.tar.gz
        rm -rf third_party_eggrollv1.tar.gz
        cd ../

        mkdir computing
        cd computing
        get_module_binary ${source_code_dir} ${module_name} eggroll-computing-${version}.tar.gz
        tar xzf eggroll-computing-${version}.tar.gz
        rm -rf eggroll-computing-${version}.tar.gz
        cd ../

        mkdir eggroll-api
        cd eggroll-api
        get_module_binary ${source_code_dir} ${module_name} eggroll-api-${version}.tar.gz
        tar xzf eggroll-api-${version}.tar.gz
        rm -rf eggroll-api-${version}.tar.gz
        cd ../

        mkdir eggroll-conf
        cd eggroll-conf
        get_module_binary ${source_code_dir} ${module_name} eggroll-conf-${version}.tar.gz
        tar xzf eggroll-conf-${version}.tar.gz
        rm -rf eggroll-conf-${version}.tar.gz
        cd ../
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
    party_label=$4
	cd ${output_packages_dir}/config/${party_label}
    cd ./${module_name}/conf
	cp ${cwd}/modify_json.py ./
	cp ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/processor-starter.sh ./

	cp ${source_code_dir}/cluster-deploy/scripts/deploy/eggroll/services.sh ./
    sed -i "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./services.sh
    sed -i "s#installdir=.*#installdir=${deploy_dir}#g" ./services.sh
	sed -i "s#PYTHONPATH=.*#PYTHONPATH=${python_path}#g" ./services.sh

    mkdir conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/${module_name}.properties ./conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/log4j2.properties ./conf
    cp  ${source_code_dir}/eggroll/framework/${module_name}/src/main/resources/applicationContext-${module_name}.xml ./conf

	sed -i "s/party.id=.*/party.id=${party_id}/g" ./conf/egg.properties
	sed -i "s/service.port=.*/service.port=${port}/g" ./conf/egg.properties
	sed -i "s/engine.names=.*/engine.names=processor/g" ./conf/egg.properties
	sed -i "s#bootstrap.script=.*#bootstrap.script=${deploy_dir}/${module_name}/processor-starter.sh#g" ./conf/egg.properties
	sed -i "s#start.port=.*#start.port=${processor_port}#g" ./conf/egg.properties
	sed -i "s#processor.venv=.*#processor.venv=${venv_dir}#g" ./conf/egg.properties
	sed -i "s#processor.python-path=.*#processor.python-path=${python_path}#g" ./conf/egg.properties
	sed -i "s#processor.engine-path=.*#processor.engine-path=${deploy_dir}/python/eggroll/computing/processor.py#g" ./conf/egg.properties
	sed -i "s#data-dir=.*#data-dir=${data_dir}#g" ./conf/egg.properties
	sed -i "s#processor.logs-dir=.*#processor.logs-dir=${deploy_dir}/logs/processor#g" ./conf/egg.properties
	sed -i "s#count=.*#count=${processor_count}#g" ./conf/egg.properties
	echo >> ./conf/egg.properties
	echo "eggroll.computing.processor.python-path=${python_path}" >> ./conf/egg.properties
}

init() {
    return 0
}

install(){
    mkdir -p ${deploy_dir}/${module_name}
    cp -r ${deploy_packages_dir}/source/${module_name}/egg-manager/* ${deploy_dir}/${module_name}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}/
    cd ${deploy_dir}/${module_name}
    ln -s eggroll-${module_name}-${version}.jar eggroll-${module_name}.jar
    mv ./services.sh ${deploy_dir}/

    cd ${deploy_packages_dir}/source/${module_name}/egg-services

    mkdir -p ${deploy_dir}/storage-service-cxx
    cp -r ./storage-service-cxx/* ${deploy_dir}/storage-service-cxx/

    mkdir -p ${deploy_dir}/python/eggroll/computing
    cp -r ./computing/* ${deploy_dir}/python/eggroll/computing/

    mkdir -p ${deploy_dir}/python/eggroll/api
    cp -r ./eggroll-api/* ${deploy_dir}/python/eggroll/api/

    mkdir -p ${deploy_dir}/python/eggroll/conf
    cp -r ./eggroll-conf/* ${deploy_dir}/python/eggroll/conf/

    cd ${deploy_dir}/storage-service-cxx
	sed -i "20s#-I. -I.*#-I. -I${deploy_dir}/storage-service-cxx/third_party/include#g" ./Makefile
	sed -i "34s#LDFLAGS += -L.*#LDFLAGS += -L${deploy_dir}/storage-service-cxx/third_party/lib -llmdb -lboost_system -lboost_filesystem -lglog -lgpr#g" ./Makefile
	sed -i "36s#PROTOC =.*#PROTOC = ${deploy_dir}/storage-service-cxx/third_party/bin/protoc#g" ./Makefile
	sed -i "37s#GRPC_CPP_PLUGIN =.*#GRPC_CPP_PLUGIN = ${deploy_dir}/storage-service-cxx/third_party/bin/grpc_cpp_plugin#g" ./Makefile
	make

    cd ${deploy_dir}/python/eggroll/conf
    cp ${deploy_dir}/${module_name}/modify_json.py ./
	#sed -i "s/clustercommip=.*/clustercommip=\"$ip\"/g" $cwd/modify_json.py
	#sed -i "s/clustercommport=.*/clustercommport=${clustercomm_port}/g" $cwd/modify_json.py
	sed -i "s/rollip=.*/rollip=\"${roll_ip}\"/g" ./modify_json.py
	sed -i "s/rollport=.*/rollport=${roll_port}/g" ./modify_json.py
	sed -i "s/proxyip=.*/proxyip=\"${proxy_ip}\"/g" ./modify_json.py
	sed -i "s/proxyport=.*/proxyport=${proxy_port}/g" ./modify_json.py
	python ./modify_json.py python ./server_conf.json
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