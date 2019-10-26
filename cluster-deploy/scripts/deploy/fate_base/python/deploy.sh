#!/bin/bash

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
set -e
module_name="python"
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
    if [[ "${deploy_mode}" == "binary" ]]; then
        get_module_binary ${source_code_dir} ${module_name} miniconda3-fate-${python_version}.tar.gz
        tar xzf miniconda3-fate-${python_version}.tar.gz
        rm -rf miniconda3-fate-${python_version}.tar.gz
    elif [[ "${deploy_mode}" == "build" ]]; then
        get_module_binary ${source_code_dir} ${module_name} miniconda3-fate-${python_version}.tar.gz
        tar xzf miniconda3-fate-${python_version}.tar.gz
        rm -rf miniconda3-fate-${python_version}.tar.gz
    fi
	return 0
}

config(){
    config_label=$4
    return 0
}

install() {
    mkdir -p ${deploy_dir}
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cd ${deploy_dir}/${module_name}/miniconda3-fate-${python_version}
    echo "#!/bin/sh
export PATH=${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin:\$PATH" > ./bin/activate
	sed -i.bak "s#!.*python#!${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin/python#g" ./bin/conda
	sed -i.bak "s#!.*python#!${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin/python#g" ./bin/conda-env
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

