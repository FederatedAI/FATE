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
source_dir=$(cd `dirname $0`; cd ../;pwd)
support_modules=(bin conf python examples fateboard eggroll proxy)
packaging_modules=()
echo ${source_dir}
if [[ -n ${1} ]]; then
    version_tag=$1
else
    version_tag="rc"
fi

cd ${source_dir}
version=`grep "FATE=" fate.env | awk -F '=' '{print $2}'`
package_dir_name="FATE_install_"${version}
package_dir=${source_dir}/cluster-deploy/${package_dir_name}
echo "[INFO] Build info"
echo "[INFO] version: "${version}
echo "[INFO] version tag: "${version_tag}
echo "[INFO] Package output dir is "${package_dir}
rm -rf ${package_dir} ${package_dir}_${version_tag}".tar.gz"
mkdir -p ${package_dir}

function packaging_bin() {
    echo "[INFO] Package bin start"
    cp -r bin ${package_dir}/
    echo "[INFO] Package bin done"
}

function packaging_conf() {
    echo "[INFO] Package conf start"
    cp fate.env RELEASE.md python/requirements.txt ${package_dir}/
    cp -r conf ${package_dir}/
    echo "[INFO] Package bin done"
}

function packaging_python(){
    echo "[INFO] Package python start"
    cp -r python ${package_dir}/
    echo "[INFO] Package python done"
}

function packaging_examples(){
    echo "[INFO] Package example start"
    cp -r examples ${package_dir}/
    echo "[INFO] Package example done"
}

packaging_fateboard(){
    echo "[INFO] Package fateboard start"
    cd ${source_dir}
    fateboard_git_url=`grep -A 3 '"fateboard"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
    fateboard_git_branch=`grep -A 3 '"fateboard"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
    echo "[INFO] Git clone fateboard submodule source code from ${fateboard_git_url} branch ${fateboard_git_branch}"
    if [[ -e "fateboard" ]];then
        while [[ true ]];do
            read -p "The fateboard directory already exists, delete and re-download? [y/n] " input
            case ${input} in
            [yY]*)
                    echo "[INFO] Delete the original fateboard"
                    rm -rf fateboard
                    git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
                    break
                    ;;
            [nN]*)
                    echo "[INFO] Use the original fateboard"
                    break
                    ;;
            *)
                    echo "Just enter y or n, please."
                    ;;
            esac
        done
    else
        git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
    fi
    cd ./fateboard
    fateboard_version=$(grep -E -m 1 -o "<version>(.*)</version>" ./pom.xml | tr -d '[\\-a-z<>//]' | awk -F "version" '{print $2}')
    echo "[INFO] fateboard version "${fateboard_version}
    mvn clean package -DskipTests
    mkdir -p ${package_dir}/fateboard/conf
    mkdir -p ${package_dir}/fateboard/ssh
    cp ./target/fateboard-${fateboard_version}.jar ${package_dir}/fateboard/
    cp ./bin/service.sh ${package_dir}/fateboard/
    cp ./src/main/resources/application.properties ${package_dir}/fateboard/conf/
    cd ${package_dir}/fateboard
    touch ./ssh/ssh.properties
    ln -s fateboard-${fateboard_version}.jar fateboard.jar
    echo "[INFO] Package fateboard done"
}

packaging_eggroll(){
    echo "[INFO] Package eggroll start"
    cd ${source_dir}
    eggroll_git_url=`grep -A 3 '"eggroll"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
    eggroll_git_branch=`grep -A 3 '"eggroll"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
    echo "[INFO] Git clone eggroll submodule source code from ${eggroll_git_url} branch ${eggroll_git_branch}"
    if [[ -e "eggroll" ]];then
        while [[ true ]];do
            read -p "The eggroll directory already exists, delete and re-download? [y/n] " input
            case ${input} in
            [yY]*)
                    echo "[INFO] Delete the original eggroll"
                    rm -rf eggroll
                    git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
                    break
                    ;;
            [nN]*)
                    echo "[INFO] Use the original eggroll"
                    break
                    ;;
            *)
                    echo "Just enter y or n, please."
                    ;;
            esac
        done
    else
        git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
    fi
    cd ./eggroll
    cd ./deploy
    sh ./auto-packaging.sh
    mkdir -p ${package_dir}/eggroll
    mv ${source_dir}/eggroll/eggroll.tar.gz ${package_dir}/eggroll/
    cd ${package_dir}/eggroll/
    tar xzf eggroll.tar.gz
    rm -rf eggroll.tar.gz
    echo "[INFO] Package eggroll done"
}

function packaging_proxy(){
    echo "[INFO] Package proxy start"
    cd ${source_dir}
    cd c/proxy
    mkdir -p ${package_dir}/proxy/nginx
    cp -r conf lua ${package_dir}/proxy/nginx
    echo "[INFO] Package proxy done"
}

compress(){
    echo "[INFO] Compress start"
    cd ${package_dir}
    touch ./packages_md5.txt
    os_kernel=`uname -s`
    for module in ${packaging_modules[@]};
    do
        tar czf ${module}.tar.gz ./${module}
        case "${os_kernel}" in
            Darwin)
                md5_value=`md5 ${module}.tar.gz | awk '{print $4}'`
                ;;
            Linux)
                md5_value=`md5sum ${module}.tar.gz | awk '{print $1}'`
                ;;
        esac
        echo "${module}:${md5_value}" >> ./packages_md5.txt
        rm -rf ./${module}
    done
    echo "[INFO] Compress done"
    echo "[INFO] A total of `ls ${package_dir} | wc -l | awk '{print $1}'` packages:"
    ls -lrt ${package_dir}
    cd ${source_dir}/cluster-deploy/
    tar czf ${package_dir_name}_${version_tag}".tar.gz" ${package_dir_name}
}


build() {
    echo "[INFO] Packaging start------------------------------------------------------------------------"
    for module in ${packaging_modules[@]};
    do
        packaging_${module}
        echo
    done
    echo "[INFO] Packaging end ------------------------------------------------------------------------"
    compress
}

all() {
    for ((i=0;i<${#support_modules[*]};i++))
    do
        packaging_modules[i]=${support_modules[i]}
	done
	build
}


multiple() {
    total=$#
    for ((i=2;i<total+1;i++)); do
        packaging_modules[i]=${!i//\//}
    done
	build
}

usage() {
    echo "usage: $0 {version_tag} {all|[module1, ...]}"
}


case "$2" in
    all)
        all $@
        ;;
    usage)
        usage
        ;;
    *)
        multiple $@
        ;;
esac