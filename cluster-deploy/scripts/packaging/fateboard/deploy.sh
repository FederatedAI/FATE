#!/bin/bash
set -e
module_name="fateboard"
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


package() {
	cd ${output_packages_dir}/source
	if [ -e "${module_name}" ]
	then
		rm -rf ${module_name}
	fi
	mkdir ${module_name}
	
	cd ${source_code_dir}/${module_name}

    ping -c 4 www.baidu.com >>/dev/null 2>&1
	if [ $? -eq 0 ]
	then
		echo "start execute mvn build"
	    mvn clean package -DskipTests
	    echo "mvn  build done"
	else
	    echo "Sorry,the host cannot access the public network."
		exit
	fi
	if [ ! -f "${source_code_dir}/${module_name}/target/${module_name}-${version}.jar" ]
	then
		echo "${source_code_dir}/${module_name}/target/${module_name}-${version}.jar: file doesn't exist."
		exit
	fi
	cp ${source_code_dir}/${module_name}/target/${module_name}-${version}.jar ${output_packages_dir}/source/${module_name}
	cd  ${output_packages_dir}/source/${module_name}
#	ln -s ${module_name}-${version}.jar ${module_name}.jar
}



config() {
    node_label=$4
	if [[ ${node_label} == "" ]]
	then
		echo "usage: $0 {apt/build} {package|config|install|init} {configurations path} {node_ip}."
		exit
	fi
	cd ${output_packages_dir}/config/${node_label}
	if [ -e "${module_name}" ]
	then
		rm -rf ${module_name}
	fi
	mkdir -p ./${module_name}/conf
	cd ./${module_name}/conf

	mkdir conf ssh 
	cp ${source_code_dir}/${module_name}/src/main/resources/application.properties ./conf
	touch ./ssh/ssh.properties
	
	cp ${source_code_dir}/cluster-deploy/example-dir-tree/${module_name}/service.sh ./
	sed -i "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./service.sh

	sed -i "s#^server.port=.*#server.port=${fateboard_port}#g" ./conf/application.properties
	sed -i "s#^fateflow.url=.*#fateflow.url=http://${fate_flow_ip}:${fate_flow_port}#g" ./conf/application.properties
	sed -i "s#^spring.datasource.driver-Class-Name=.*#spring.datasource.driver-Class-Name=com.mysql.cj.jdbc.Driver#g" ./conf/application.properties
	sed -i "s#^spring.datasource.url=.*#spring.datasource.url=jdbc:mysql://${db_ip}:3306/${db_name}?characterEncoding=utf8\&characterSetResults=utf8\&autoReconnect=true\&failOverReadOnly=false\&serverTimezone=GMT%2B8#g" ./conf/application.properties
	sed -i "s/^spring.datasource.username=.*/spring.datasource.username=${db_user}/g" ./conf/application.properties
	sed -i "s/^spring.datasource.password=.*/spring.datasource.password=${db_password}/g" ./conf/application.properties
	for node in "${node_list[@]}"
	do
		echo ${node}
		node_info=(${node})
		sed -i "/${node_info[0]}/d" ./ssh/ssh.properties
		echo "${node_info[0]}=${node_info[1]}|${node_info[2]}|${node_info[3]}" >> ./ssh/ssh.properties
	done
	cd ..
	cp ${cwd}/deploy.sh ./
    cp ${cwd}/${config_path} ./configurations.sh
}


install() {
    mkdir -p ${deploy_dir}/
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
	
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
	cd ${deploy_dir}/${module_name}
	ln -s ${module_name}-${version}.jar ${module_name}.jar
}

init() {
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
