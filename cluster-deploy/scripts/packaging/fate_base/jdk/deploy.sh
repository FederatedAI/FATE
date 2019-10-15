#!/bin/bash
set -e
module_name="jdk"
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

# deploy functions

package() {
    if [[ "${deploy_mode}" == "apt" ]]; then
        cd ${output_packages_dir}/source
        if [[ -e "${module_name}" ]]
        then
            rm ${module_name}
        fi
        mkdir -p ${module_name}
        cd ${module_name}
        cp ${source_code_dir}/cluster-deploy/scripts/fate-base/packages/jdk-${jdk_version}-linux-x64.tar.gz ./
        tar xzf jdk-${jdk_version}-linux-x64.tar.gz
        rm -rf jdk-${jdk_version}-linux-x64.tar.gz
        mkdir tmp
        cp -r jdk*/* tmp
        rm -rf jdk*
        mv tmp jdk-${jdk_version}
    elif [[ "${deploy_mode}" == "build" ]]; then
        echo "not support"
    fi
	return 0
}

config(){
    node_label=$4
	cd ${output_packages_dir}/config/${node_label}
	if [[ -e "${module_name}" ]]
	then
		rm ${module_name}
	fi
	mkdir -p ./${module_name}/conf

	cd ${module_name}
    cp ${cwd}/deploy.sh ./
    cp ${cwd}/${config_path} ./configurations.sh
    return 0
}

install() {
    mkdir -p ${deploy_dir}
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
}

init(){
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


