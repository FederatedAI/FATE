#!/bin/bash
set -e
module_name="jdk"
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

packaging() {
    source ../../../default_configurations.sh
    package_init ${output_packages_dir} ${module_name}
    get_module_binary ${source_code_dir} ${module_name} jdk-${jdk_version}-linux-x64.tar.gz
    tar xzf jdk-${jdk_version}-linux-x64.tar.gz
    rm -rf jdk-${jdk_version}-linux-x64.tar.gz
    mkdir tmp
    cp -r jdk*/* tmp
    rm -rf jdk*
    mv tmp jdk-${jdk_version}
	return 0
}

config(){
    config_label=$4
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


