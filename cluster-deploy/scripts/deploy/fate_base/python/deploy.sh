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
    pip_env_packaging
	return 0
}

pip_env_packaging() {
    get_module_package ${source_code_dir} "${module_name} miniconda" Miniconda3-4.5.4-Linux-x86_64.sh
    get_module_package ${source_code_dir} "${module_name} pip packages" pip-packages-fate-${python_version}.tar.gz
    tar xzf pip-packages-fate-${python_version}.tar.gz
    rm -rf pip-packages-fate-${python_version}.tar.gz
    cp ${source_code_dir}/requirements.txt ./
}

conda_env_packaging() {
    get_module_package ${source_code_dir} ${module_name} miniconda3-fate-${python_version}.tar.gz
    tar xzf miniconda3-fate-${python_version}.tar.gz
    rm -rf miniconda3-fate-${python_version}.tar.gz
}

config(){
    config_label=$4
    return 0
}

install() {
    mkdir -p ${deploy_dir}
    pip_env_install
}

pip_env_install() {
    miniconda3_dir=${deploy_dir}/python/miniconda3
    venv_dir=${deploy_dir}/python/venv
    rm -rf ${miniconda3_dir}
    rm -rf ${venv_dir}
    cd ${deploy_packages_dir}/source/${module_name}
    sh ./Miniconda3-*-Linux-x86_64.sh -b -p ${miniconda3_dir}
    ${miniconda3_dir}/bin/pip install virtualenv -f ./pip-packages-fate-${python_version} --no-index
    ${miniconda3_dir}/bin/virtualenv -p ${miniconda3_dir}/bin/python3.6  --no-wheel --no-setuptools --no-download ${venv_dir}
    source ${venv_dir}/bin/activate
    pip install ./pip-packages-fate-${python_version}/setuptools-41.4.0-py2.py3-none-any.whl
    pip install -r ./requirements.txt -f ./pip-packages-fate-${python_version} --no-index
    pip list | wc -l
}

conda_env_install() {
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cd ${deploy_dir}/${module_name}/miniconda3-fate-${python_version}
    echo "#!/bin/sh
export PATH=${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin:\$PATH" > ./bin/activate
	sed -i.bak "s#!.*python#!${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin/python#g" ./bin/conda
	sed -i.bak "s#!.*python#!${deploy_dir}/${module_name}/miniconda3-fate-${python_version}/bin/python#g" ./bin/conda-env
	rm -rf ./bin/conda.bak ./bin/conda-env.bak
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

