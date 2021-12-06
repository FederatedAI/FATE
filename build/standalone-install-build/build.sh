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
source_dir=$(
    cd $(dirname $0)
    cd ../
    cd ../
    pwd
)
echo "[INFO] source dir: ${source_dir}"
cd ${source_dir}
source ./bin/common.sh

support_modules=(bin conf examples fate fateflow fateboard)
environment_modules=(python36 jdk pypi)

if [[ -n ${1} ]]; then
    version_tag=$1
else
    version_tag="rc"
fi
if_skip_compress=$2

version=$(grep "FATE=" fate.env | awk -F '=' '{print $2}')
install_package_dir_name="FATE_install_${version}_${version_tag}"
install_package_dir=${source_dir}/${install_package_dir_name}
package_dir_name="standalone_fate_install_${version}_${version_tag}"
package_dir=${source_dir}/${package_dir_name}
echo "[INFO] build info"
echo "[INFO] version: "${version}
echo "[INFO] version tag: "${version_tag}
echo "[INFO] package output dir is "${package_dir}
rm -rf ${package_dir} ${package_dir}".tar.gz"
mkdir -p ${package_dir}

echo "${environment_modules[@]}" "${support_modules[@]}"

function packaging() {
    echo "[INFO] package start"
    cd ${source_dir}

    # init.sh
    cp build/standalone-install-build/init.sh ${package_dir}
    fate_sed_cmd "s/version=.*/version=${version}/g" ${package_dir}/init.sh

    echo "[INFO] get install packages"
    if [[ -d ${install_package_dir} ]]; then
        echo "[INFO] install package already exists, skip build"
    else
        bash build/package-build/build.sh ${version_tag} "${environment_modules[@]}" "${support_modules[@]}"
    fi
    echo "[INFO] get install packages done"

    cp -r ${install_package_dir}/* ${package_dir}/
    echo "[INFO] copy install packages done"

    cd ${package_dir}
    echo "[INFO] deal env packages"
    env_dir=${package_dir}/env
    if [[ -d "${env_dir}" ]]; then
        rm -rf ${env_dir}
    fi
    mkdir -p ${env_dir}

    for module in "${environment_modules[@]}"; do
        tar xzf ${module}.tar.gz -C ${env_dir}
    done
    echo "[INFO] deal env packages done"

    echo "[INFO] deal system packages"
    for module in "${support_modules[@]}"; do
        tar xzf ${module}.tar.gz
        rm -rf ${module}.tar.gz
    done
    rm -rf *.tar.gz
    echo "[INFO] deal system packages done"

    echo "[INFO] package done"
}

compress() {
    echo "[INFO] compress start"
    echo "[INFO] a total of $(ls ${package_dir} | wc -l | awk '{print $1}') packages:"
    ls -lrt ${package_dir}
    package_dir_parent=$(
        cd $(dirname ${package_dir})
        pwd
    )
    cd ${package_dir_parent}
    if [[ ${if_skip_compress} -eq 1 ]]; then
        echo "[INFO] skip compress"
    else
        tar czf ${package_dir_name}".tar.gz" ${package_dir_name}
    fi
    echo "[INFO] compress done"
}

build() {
    echo "[INFO] packaging start------------------------------------------------------------------------"
    packaging
    echo "[INFO] packaging end ------------------------------------------------------------------------"
    compress
}

usage() {
    echo "usage: $0 {version_tag} {if_skip_compress}"
}

case "$2" in
usage)
    usage
    ;;
*)
    build
    ;;
esac
