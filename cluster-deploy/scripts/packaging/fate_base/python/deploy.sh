#!/bin/bash
set -e
module_name="python"
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
    source ../../../default_configurations.sh
    if [[ "${deploy_mode}" == "apt" ]]; then
        cd ${output_packages_dir}/source
        if [[ -e "${module_name}" ]]
        then
            rm ${module_name}
        fi
        mkdir -p ${module_name}
        cd ${module_name}
        copy_path=${source_code_dir}/cluster-deploy/packages/miniconda3-fate-${python_version}.tar.gz
        download_uri=${fate_cos_address}/miniconda3-fate-${python_version}.tar.gz
        if [[ -f ${copy_path} ]];then
            echo "[INFO] Copying ${copy_path}"
            cp ${copy_path} ./
        else
            echo "[INFO] Downloading ${download_uri}"
            wget ${download_uri}
        fi
        tar xzf miniconda3-fate-${python_version}.tar.gz
        rm -rf miniconda3-fate-${python_version}.tar.gz
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

	cd ${output_packages_dir}/config/${node_label}/${module_name}
    cp ${cwd}/deploy.sh ./
    cp ${cwd}/${config_path} ./configurations.sh
    return 0
}

install() {
    mkdir -p ${deploy_dir}
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cd ${deploy_dir}/${module_name}/miniconda3-fate-${python_version}
    echo "#!/bin/sh
export PATH=${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin:\$PATH" > ./bin/activate
	sed -i "s#!.*python#!${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin/python#g" ./bin/conda
	sed -i "s#!.*python#!${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin/python#g" ./bin/conda-env
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

