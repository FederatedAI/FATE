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
    get_module_package ${source_code_dir} ${module_name} redis-${redis_version}.tar.gz
    tar xzf redis-${redis_version}.tar.gz
    rm -rf redis-${redis_version}.tar.gz
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
    rm -rf ./redis.conf.bak
    return 0
}

install () {
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cd ${deploy_dir}/${module_name}/redis-${redis_version}
    make
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}/redis-${redis_version}
    sh service.sh stop
    mkdir bin
    cp ./src/redis-server ./bin
    cp ./src/redis-cli ./bin
    cp ./src/redis-check-aof ./bin
    cp ./src/redis-check-rdb ./bin
    cp ./src/redis-sentinel ./bin
    cp ./src/redis-benchmark ./bin
}

init(){
    cd ${deploy_dir}/${module_name}/redis-${redis_version}
    sh service.sh restart
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
