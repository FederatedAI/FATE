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
#input_dir=${fateboard_input_dir}
#output_dir=${fateboard_output_dir}
#version=${fateboard_version}
#java_dir=${javadir}
#fbport=${fateboard_port}
#flip=${fateflow_ip}
#fldbip=${fateflow_db[0]}
#fldbname=${fateflow_db[1]}
#fldbuser=${fateflow_db[2]}
#fldbpasswd=${fateflow_db[3]}	
#iplist=${iplist}
#output_dir=${fate_dir}/cluster-deploy/example-dir-tree

source_build() {
	#if [ ! -d "${input_dir}" ]
	#then
	#	echo "${input_dir}: directory doesn't exist."
	#	exit
	#fi
	#cd ${input_dir}/fateboard
	cd ${fate_dir}/cluster-deploy/example-dir-tree/proxy
	if [ -f "fate-proxy-${version}.tar.gz" ]
	then
		rm fate-proxy-${version}.tar.gz
	fi
	cd ${fate_dir}/arch

    ping -c 4 www.baidu.com >>/dev/null 2>&1
	if [ $? -eq 0 ]
	then
		echo "start execute mvn build"
	    mvn clean package -DskipTests
	    echo "mvn  build done"
	else
	    echo "Sorry,the host cannot access the public network."
	fi
	
	if [ ! -f "networking/proxy/target/fate-proxy-${version}.jar" ]
	then
		echo "${fate_dir}/arch/networking/proxy/target/fate-proxy-${version}.jar: file doesn't exist."
		exit
	fi
	
	cp ${fate_dir}/arch/networking/proxy/target/fate-proxy-${version}.jar ${fate_dir}/cluster-deploy/example-dir-tree/proxy
	cp -r ${fate_dir}/arch/networking/proxy/target/lib ${fate_dir}/cluster-deploy/example-dir-tree/proxy
	cd ${fate_dir}/cluster-deploy/example-dir-tree/proxy
	ln -s fate-proxy-${version}.jar fate-proxy.jar
	tar -czf fate-proxy-${version}.tar.gz ./fate-proxy-${version}.jar ./fate-proxy.jar ./lib
	rm -rf fate-proxy-${version}.jar fate-proxy.jar ./lib
	
}

build() {
	#cd ${fate_dir}/cluster-deploy/example-dir-tree/proxy 
	#wget 
	return 0 
}

config() {
#	if [ ! -d "${output_dir}" ]
#	then
#		echo "${output_dir}: directory doesn't exist."
#		exit
#	fi
	cd ${fate_dir}/cluster-deploy/example-dir-tree/proxy
	if [ -f "fate-proxy-config-${version}.tar.gz" ]
	then
		rm fate-proxy-config-${version}.tar.gz
	fi
	
	if [  -d "conf" ]
	then
		rm -rf conf
		
	fi

	mkdir conf
	cp ${fate_dir}/arch/networking/proxy/src/main/resources/applicationContext-proxy.xml ./conf
	cp ${fate_dir}/arch/networking/proxy/src/main/resources/log4j2.properties ./conf
	cp ${fate_dir}/arch/networking/proxy/src/main/resources/proxy.properties ./conf
	cp ${fate_dir}/arch/networking/proxy/src/main/resources/route_tables/route_table.json ./conf
	
	sed -i "s#JAVA_HOME=.*#JAVA_HOME=${javadir}#g" ./service.sh

	
	export PYTHONPATH=${PYTHONPATH}
	source $venvdir/bin/activate
	sed -i "s/port=.*/port=${port}/g" ./conf/proxy.properties
	sed -i "s#route.table=.*#route.table=${dir}/proxy/conf/route_table.json#g" ./conf/proxy.properties
	sed -i "s/coordinator=.*/coordinator=${partyid}/g" ./conf/proxy.properties
	sed -i "s/ip=.*/ip=${pip}/g" ./conf/proxy.properties
	
	sed -i "s/exchangeip=.*/exchangeip=\"${exchangeip}\"/g" ./proxy_modify_json.py
	sed -i "s/fip=.*/fip=\"${fip}\"/g" ./proxy_modify_json.py
	sed -i "s/flip=.*/flip=\"${flip}\"/g" ./proxy_modify_json.py
	sed -i "s/sip1=.*/sip1=\"${sip1}\"/g" ./proxy_modify_json.py
	sed -i "s/sip2=.*/sip2=\"${sip2}\"/g" ./proxy_modify_json.py
	sed -i "s/partyId=.*/partyId=\"${partyid}\"/g" ./proxy_modify_json.py
	
	python proxy_modify_json.py proxy ./conf/route_table.json
	
	tar -czf  fate-proxy-config-${version}.tar.gz ./conf ./proxy_modify_json.py ./service.sh
	rm -rf  ./conf 
	
#	mv fateboard.tar.gz ${output_dir}

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

# 1. moudle 2.config_path