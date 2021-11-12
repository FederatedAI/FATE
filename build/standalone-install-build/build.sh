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
source_dir=$(cd `dirname $0`; cd ../;cd ../;pwd)
support_modules=(bin conf examples fate fateflow fateboard)
environment_modules=(python36 jdk pypi)
echo ${source_dir}
if [[ -n ${1} ]]; then
    version_tag=$1
else
    version_tag="rc"
fi

cd ${source_dir}
echo "[INFO] source dir: ${source_dir}"
version=`grep "FATE=" fate.env | awk -F '=' '{print $2}'`
fate_packages_dir_name="FATE_install_"${version}
fate_packages_dir=${source_dir}/${fate_packages_dir_name}
package_dir_name="standalone_fate_install_"${version}
package_dir=${source_dir}/${package_dir_name}
echo "[INFO] build info"
echo "[INFO] version: "${version}
echo "[INFO] version tag: "${version_tag}
echo "[INFO] package output dir is "${package_dir}
rm -rf ${package_dir} ${package_dir}_${version_tag}".tar.gz"
mkdir -p ${package_dir}

function packaging_env(){
    echo "[INFO] package env start"
    cd ${source_dir} 
    cp build/standalone-install-build/init.sh ${package_dir}
    sed -i.bak "s/version=.*/version=${version}/g" ${package_dir}/init.sh

    echo "[INFO] enter build packages"
    sh build/package-build/build.sh ${version_tag} "${environment_modules[@]}";
    echo "[INFO] exit build packages"

    env_dir=${package_dir}/env
    if [[ -d "${env_dir}" ]];then
      rm -rf ${env_dir}
    fi
    mkdir -p ${env_dir}
    cp -r ${fate_packages_dir}/* ${env_dir}/
    
    cd ${env_dir}
    for module in "${environment_modules[@]}";
    do
        tar xzf ${module}.tar.gz
        rm -rf ${module}.tar.gz
    done
    echo "[INFO] package env done"
}

function packaging_systems(){
    echo "[INFO] package systems start"
    cd ${source_dir}
    echo "[INFO] enter build packages"
    sh build/package-build/build.sh ${version_tag} "${support_modules[@]}"
    echo "[INFO] exit build packages"
    cp -r ${fate_packages_dir}/* ${package_dir}/
    cd ${package_dir}
    for module in "${support_modules[@]}";
    do
        tar xzf ${module}.tar.gz
        rm -rf ${module}.tar.gz
    done
    echo "[INFO] package systems done"
}

compress(){
    echo "[INFO] compress start"
    echo "[INFO] a total of `ls ${package_dir} | wc -l | awk '{print $1}'` packages:"
    ls -lrt ${package_dir}
    package_dir_parent=$(cd `dirname ${package_dir}`; pwd)
    cd ${package_dir_parent}
    tar czf ${package_dir_name}_${version_tag}".tar.gz" ${package_dir_name}
    echo "[INFO] compress done"
}


build() {
    echo "[INFO] packaging start------------------------------------------------------------------------"
    packaging_env
    packaging_systems
    echo "[INFO] packaging end ------------------------------------------------------------------------"
    compress
}

usage() {
    echo "usage: $0 {version_tag}"
}


case "$2" in
    usage)
        usage
        ;;
    *)
        build
        ;;
esac