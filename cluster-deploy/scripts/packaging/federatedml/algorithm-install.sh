#!/bin/bash
set -e
config_path=$2
if [[ ${config_path} == "" ]] || [[ ! -f ${config_path} ]]
then
	echo "usage: $0 {install} {configurations path}."
	exit
fi
source ${config_path}

cwd=$(cd `dirname $0`; pwd)
cd ../../
fate_dir=`pwd`


source_build() {
	#echo ${output_dir}
	#if [ ! -d "${output_dir}" ]
	#then
	#	echo "${output_dir}: directory doesn't exist."
	#	exit
	#fi
	cd ${fate_dir}/cluster-deploy/example-dir-tree/python
	if [ -f "algorithm-${version}.tar.gz" ]
	then
		rm algorithm-${version}.tar.gz
	fi
	cd ${fate_dir}

	#cp -r arch federatedml workflow examples ${fate_dir}/cluster-deploy/example-dir-tree/python
	#cd ${fate_dir}/cluster-deploy/example-dir-tree/python
	#sed -i "s/eggroll_meta/$jdbcdbname/g" $dir/python/arch/eggroll/meta-service/src/main/resources/create-meta-service.sql

	#tar -czf  algorithm.tar.gz ./arch ./federatedml ./workflow ./examples
	#mv algorithm.tar.gz ${output_dir}
	tar -czf algorithm-${version}.tar.gz arch federatedml workflow examples
	mv algorithm-${version}.tar.gz ${fate_dir}/cluster-deploy/example-dir-tree/python
}
build() {
	#cd ${cwd}
	#wget
	#mv algorithm.tar.gz ${fate_dir}/cluster-deploy/example-dir-tree/python
	
	return 0


}

config() {
	return 0
}

init (){
	return 0
}

install(){
	source_build
	config
	init
}

case "$1" in
    source_build)
        source_build 
        ;;
    build)
        build 
        ;;
    config)
        config
        ;;
    init)
        init 
        ;;
	install)
		install
		;;
	*)
		echo "usage: $0 {source_build|build|config|init|install} {configurations path}."
        exit -1
esac

# 1.module 2.output_dir