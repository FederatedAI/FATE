#!/bin/bash
set -e
module_name="redis"
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
packaging(){
    source ../../../default_configurations.sh
    package_init ${output_packages_dir} ${module_name}
    if [[ "${deploy_mode}" == "binary" ]]; then
        get_module_binary ${source_code_dir} ${module_name} redis-${redis_version}.tar.gz
        tar xzf redis-${redis_version}.tar.gz
        rm -rf redis-${redis_version}.tar.gz
    elif [[ "${deploy_mode}" == "build" ]]; then
        echo "not support"
    fi
	return 0
}

config(){
    config_label=$4
	cd ${output_packages_dir}/config/${config_label}
	cd ./${module_name}/conf
    cp ${output_packages_dir}/source/${module_name}/redis-${redis_version}/redis.conf ./
	cp ${cwd}/service.sh ./
	sed -i.bak "s/bind 127.0.0.1/bind 0.0.0.0/g" ./redis.conf
    sed -i.bak "s/# requirepass foobared/requirepass ${redis_password}/g" ./redis.conf
    sed -i.bak "s/databases 16/databases 50/g" ./redis.conf
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
