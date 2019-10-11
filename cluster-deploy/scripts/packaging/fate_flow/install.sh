#!/bin/bash
set -e
module_name="fate_flow"
cwd=$(cd `dirname $0`; pwd)
cd $cwd
source ./configurations.sh

config_path=$2
if [[ ${config_path} == "" ]] || [[ ! -f ${config_path} ]]
then
	echo "usage: $0 {install} {configurations path}."
	exit
fi
source ${config_path}

package_source() {
    cd ${packages_dir}
	if [ -f "fate_flow-${version}.tar.gz" ]
	then
		rm fate_flow-${version}.tar.gz
	fi
	cp -r ${source_dir}/fate_flow $packages_dir
	tar -czf fate_flow-${version}.tar.gz  ./fate_flow
	rm -rf ./fate_flow
	return 0
}

source_build() {
    echo "[INFO][$module_name] build from source code"
	return 0
}

build() {
	return 0
}

config() {
	cd ${packages_dir}
	if [ ! -f "fate_flow-${version}.tar.gz" ]
	then
		echo "fate_flow-${version}.tar.gz doesn't exist."
		return 1
	fi
	if [ -f "fate_flow-${version}-config.tar.gz" ]
	then
		rm fate_flow-${version}-config.tar.gz
	fi
	
	tar -xvf fate_flow-${version}.tar.gz
	if [[ ! -f "fate_flow/service.sh" ]] || [[ ! -f "fate_flow/settings.py" ]]
		then
		echo "[ERROR][$module_name] can not found fate_flow/service.sh or fate_flow/settings.py"
		return 2
	fi
	
	sed -i "s#PYTHONPATH=.*#PYTHONPATH=${deploy_dir}/python#g" ./fate_flow/service.sh
	sed -i "s#venv=.*#venv=${venv_dir}#g" ./fate_flow/service.sh
	
	sed -i "s/WORK_MODE =.*/WORK_MODE = 1/g" ./fate_flow/settings.py
	sed -i "s/'user':.*/'user': '${db_user}',/g" ./fate_flow/settings.py
	sed -i "s/'passwd':.*/'passwd': '${db_password}',/g" ./fate_flow/settings.py
	sed -i "s/'host':.*/'host': '${db_ip}',/g" ./fate_flow/settings.py
	sed -i "s/'name':.*/'name': '${db_name}',/g" ./fate_flow/settings.py
	sed -i "s/'password':.*/'password': '${redis_password}',/g" ./fate_flow/settings.py
	sed "/'host':.*/{x;s/^/./;/^\.\{2\}$/{x;s/.*/    'host': '${redis_ip}',/;x};x;}" ./fate_flow/settings.py
	
	tar -czf  fate_flow-${version}-config.tar.gz ./fate_flow
	rm -rf ./source_dir
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
    package_source)
        package_source
        ;;
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
