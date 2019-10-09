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
	if [ -f "fate_flow-${version}.tar.gz" ]
	then
		rm fate_flow-${version}.tar.gz
	fi
	cp -r ${fate_dir}/fate_flow ${fate_dir}/cluster-deploy/example-dir-tree/python
	tar -czf fate_flow-${version}.tar.gz  ./fate_flow
	rm -rf ./fate_flow
}

build() {
	#cd ${fate_dir}/cluster-deploy/example-dir-tree/python 
	#wget 
	return 0 
}

config() {
	cd ${fate_dir}/cluster-deploy/example-dir-tree/python
	
	if [ ! -f "fate_flow-${version}.tar.gz" ]
	then
		echo "fate_flow-${version}.tar.gz doesn't exist."
		exit
	fi
	if [ -f "fate_flow-${version}-config.tar.gz" ]
	then
		rm fate_flow-${version}-config.tar.gz
	fi
	
	tar -xvf fate_flow-${version}.tar.gz
	if [[ ! -f "fate_flow/service.sh" ]] || [[ ! -f "fate_flow/settings.py" ]]
		then
		echo "usage: $0 {install} {configurations path}."
		exit
	fi
	
	sed -i "s#PYTHONPATH=.*#PYTHONPATH=${dir}/python#g" ./fate_flow/service.sh
	sed -i "s#venv=.*#venv=${venvdir}#g" ./fate_flow/service.sh
	
	sed -i "s/WORK_MODE =.*/WORK_MODE = 1/g" ./fate_flow/settings.py
	sed -i "s/PARTY_ID =.*/PARTY_ID = \"${partyid}\"/g" ./fate_flow/settings.py
	sed -i "s/'user':.*/'user': '${fldbuser}',/g" ./fate_flow/settings.py
	sed -i "s/'passwd':.*/'passwd': '${fldbpasswd}',/g" ./fate_flow/settings.py
	sed -i "s/'host':.*/'host': '${fldbip}',/g" ./fate_flow/settings.py
	sed -i "s/'name':.*/'name': '${fldbname}',/g" ./fate_flow/settings.py
	sed -i "s/localhost/${flip}/g" ./fate_flow/settings.py
	sed -i "s/'password':.*/'password': '${redispass}',/g" ./fate_flow/settings.py
	sed "/'host':.*/{x;s/^/./;/^\.\{2\}$/{x;s/.*/    'host': '${redisip}',/;x};x;}" ./fate_flow/settings.py
	
	tar -czf  fate_flow-${version}-config.tar.gz ./fate_flow
	rm -rf ./fate_dir

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
