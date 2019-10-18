#!/bin/bash

set -e
module_name="federation"
cwd=$(cd `dirname $0`; pwd)

deploy_mode=$1
config_path=$3

usage() {
	echo "usage: $0 {apt/build} {package|config|install|init} {configurations path}."
}

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
	
	cd ${source_code_dir}/arch
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
	if [ ! -f "${source_code_dir}/arch/driver/${module_name}/target/fate-${module_name}-${version}.jar" ]
	then
		echo "${source_code_dir}/arch/driver/${module_name}/target/fate-${module_name}-${version}.jar: file doesn't exist."
		exit
	fi
	cp ${source_code_dir}/arch/driver/${module_name}/target/fate-${module_name}-${version}.jar ${output_packages_dir}/source/${module_name}
	cp -r ${source_code_dir}/arch/driver/${module_name}/target/lib ${output_packages_dir}/source/${module_name}

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
	mkdir conf

	cp  ${source_code_dir}/arch/driver/${module_name}/src/main/resources/federation.properties ./conf
	cp  ${source_code_dir}/arch/driver/${module_name}/src/main/resources/log4j2.properties ./conf
	cp  ${source_code_dir}/arch/driver/${module_name}/src/main/resources/applicationContext-federation.xml ./conf

	sed -i "s/service.port=.*/service.port=${federation_port}/g" ./conf/federation.properties
	sed -i "s/meta.service.port=.*/meta.service.port=${meta_service_port}/g" ./conf/federation.properties
	sed -i "s/party.id=.*/party.id=${party_id}/g" ./conf/federation.properties
	sed -i "s/meta.service.ip=.*/meta.service.ip=${meta_service_ip}/g" ./conf/federation.properties
	
	cp ${source_code_dir}/cluster-deploy/example-dir-tree/${module_name}/service.sh ./
	sed -i "s#JAVA_HOME=.*#JAVA_HOME=${java_dir}#g" ./service.sh
	
	cd ../
	cp ${cwd}/deploy.sh ./
    cp ${cwd}/${config_path} ./configurations.sh
}
init() {
	return 0
}

install(){
    mkdir -p ${deploy_dir}/
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
	
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}
	cd ${deploy_dir}/${module_name}
	ln -s fate-${module_name}-${version}.jar ${module_name}.jar
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

