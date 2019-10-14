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
	cd ${fate_dir}/cluster-deploy/example-dir-tree/fateboard
	if [ -f "fateboard-${version}.tar.gz" ]
	then
		rm fateboard-${version}.tar.gz
	fi
	cd ${fate_dir}/fateboard

    ping -c 4 www.baidu.com >>/dev/null 2>&1
	if [ $? -eq 0 ]
	then
		echo "start execute mvn build"
	    mvn clean package -DskipTests
	    echo "mvn  build done"
	else
	    echo "Sorry,the host cannot access the public network."
	fi
	if [ ! -f "target/fateboard-${version}.jar" ]
	then
		echo "${fate_dir}/fateboard/target/fateboard-${version}.jar: file doesn't exist."
		exit
	fi
	cp ${fate_dir}/fateboard/target/fateboard-${version}.jar ${fate_dir}/cluster-deploy/example-dir-tree/fateboard
	cd ${fate_dir}/cluster-deploy/example-dir-tree/fateboard
	ln -s fateboard-$version.jar fateboard.jar
	tar -czf fateboard-${version}.tar.gz ./fateboard-$version.jar ./fateboard.jar
	rm fateboard-$version.jar fateboard.jar
	
}

build() {
	#cd ${fate_dir}/cluster-deploy/example-dir-tree/fateboard 
	#wget 
	return 0 
}

config() {
#	if [ ! -d "${output_dir}" ]
#	then
#		echo "${output_dir}: directory doesn't exist."
#		exit
#	fi
	cd ${fate_dir}/cluster-deploy/example-dir-tree/fateboard
	if [ -f "fateboard-${version}-config.tar.gz" ]
	then
		rm fateboard-${version}-config.tar.gz
	fi

	
	mkdir conf ssh 
	cp ${fate_dir}/fateboard/src/main/resources/application.properties ./conf
	touch ./ssh/ssh.properties
	
	sed -i "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./service.sh

	sed -i "s#^server.port=.*#server.port=${fbport}#g" ./conf/application.properties
	sed -i "s#^fateflow.url=.*#fateflow.url=http://${flip}:${flport}#g" ./conf/application.properties
	sed -i "s#^spring.datasource.driver-Class-Name=.*#spring.datasource.driver-Class-Name=com.mysql.cj.jdbc.Driver#g" ./conf/application.properties
	sed -i "s#^spring.datasource.url=.*#spring.datasource.url=jdbc:mysql://${fldbip}:3306/${fldbname}?characterEncoding=utf8\&characterSetResults=utf8\&autoReconnect=true\&failOverReadOnly=false\&serverTimezone=GMT%2B8#g" ./conf/application.properties
	sed -i "s/^spring.datasource.username=.*/spring.datasource.username=${fldbuser}/g" ./conf/application.properties
	sed -i "s/^spring.datasource.password=.*/spring.datasource.password=${fldbpasswd}/g" ./conf/application.properties
	for node in "${nodelist[@]}"
	do
		echo ${node}
		node_info=(${node})
		sed -i "/${node_info[0]}/d" ./ssh/ssh.properties
		echo "${node_info[0]}=${node_info[1]}|${node_info[2]}|${node_info[3]}" >> ./ssh/ssh.properties
	done

	tar -czf  fateboard-${version}-config.tar.gz ./ssh ./conf ./service.sh
	rm -rf  ./ssh ./conf 
	
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

# 1.moudle 2.input_dir 3.output_dir 4.version 5.java_dir 6.flip 7.fldbip 8.fldbname 9.fldbuser/g 10.fldbpasswd/g 11.iplist
# 1. moudle 2.config_path