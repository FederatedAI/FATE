#!/bin/bash
set -e
module_name="redis"
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./configurations.sh

usage() {
	echo "usage: $0 {apt/build} {package|config|install|init} {configurations path}."
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
package(){
    source ../../../default_configurations.sh
    if [[ "${deploy_mode}" == "apt" ]]; then
        cd ${output_packages_dir}/source
        if [[ -e "${module_name}" ]]
        then
            rm ${module_name}
        fi
        mkdir -p ${module_name}
        cd ${module_name}
        copy_path=${source_code_dir}/cluster-deploy/packages/redis-${redis_version}.tar.gz
        download_uri=${fate_cos_address}/redis-${redis_version}.tar.gz
        if [[ -f ${copy_path} ]];then
            echo "[INFO] Copying ${copy_path}"
            cp ${copy_path} ./
        else
            echo "[INFO] Downloading ${download_uri}"
            wget ${fate_cos_address}/redis-${redis_version}.tar.gz
        fi
        tar xzf redis-${redis_version}.tar.gz
        rm -rf redis-${redis_version}.tar.gz
    elif [[ "${deploy_mode}" == "build" ]]; then
        echo "not support"
    fi
	return 0
}

config(){
    node_label=$4
	cd ${output_packages_dir}/config/${node_label}
	if [[ -e "${module_name}" ]]
	then
		rm ${module_name}
	fi
	mkdir -p ./${module_name}/conf

	cd ./${module_name}/conf
    cp ${output_packages_dir}/source/${module_name}/redis-${redis_version}/redis.conf ./
	cp ${cwd}/service.sh ./
	sed -i "s/bind 127.0.0.1/bind 0.0.0.0/g" ./redis.conf
    sed -i "s/# requirepass foobared/requirepass ${redis_password}/g" ./redis.conf
    sed -i "s/databases 16/databases 50/g" ./redis.conf

	cd ../
    cp ${cwd}/deploy.sh ./
    cp ${cwd}/${config_path} ./configurations.sh
    return 0
}

install () {
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cd ${deploy_dir}/${module_name}/redis-${redis_version}
    make
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}/redis-${redis_version}
    mkdir bin
    cp ./src/redis-server ./bin
    cp ./src/redis-cli ./bin
    cp ./src/redis-check-aof ./bin
    cp ./src/redis-check-rdb ./bin
    cp ./src/redis-sentinel ./bin
    cp ./src/redis-benchmark ./bin
}

init(){
    return 0
}

case "$2" in
    package)
        package $*
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
