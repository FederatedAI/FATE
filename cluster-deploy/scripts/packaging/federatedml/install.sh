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
	cd ${fate_dir}/cluster-deploy/example-dir-tree/python
	if [ -f "algorithm-${version}.tar.gz" ]
	then
		rm algorithm-${version}.tar.gz
	fi
	cd ${fate_dir}
	tar -czf algorithm-${version}.tar.gz arch federatedml workflow examples
	mv algorithm-${version}.tar.gz ${fate_dir}/cluster-deploy/example-dir-tree/python
}
build() {
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
