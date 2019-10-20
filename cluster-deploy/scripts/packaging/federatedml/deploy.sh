#!/bin/bash
set -e
module_name="federatedml"
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./configurations.sh

usage() {
	echo "usage: $0 {binary/build} {package|config|install|init} {configurations path}."
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
    cd ${output_packages_dir}/source
	if [[ -e "${module_name}" ]]
	then
		rm ${module_name}
	fi
	mkdir -p ${module_name}
	cp -r ${source_code_dir}/federatedml ${output_packages_dir}/source/${module_name}/
	cp -r ${source_code_dir}/examples ${output_packages_dir}/source/${module_name}
	mkdir -p ${output_packages_dir}/source/${module_name}/arch
	cp -r ${source_code_dir}/arch/api ${output_packages_dir}/source/${module_name}/arch/
	return 0
}

config() {
    node_label=$4
	return 0
}

install () {
    mkdir -p ${deploy_dir}
    cp -r ${deploy_packages_dir}/source/${module_name}/* ${deploy_dir}/
}

init (){
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
