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
support_modules=(bin conf python examples fateboard eggroll)
packaging_modules=()
echo ${source_dir}
if [[ -n ${1} ]]; then
    version_tag=$1
else
    version_tag="rc"
fi

cd ${source_dir}
echo "[INFO] source dir: ${source_dir}"
#git submodule foreach --recursive git pull
version=`grep "FATE=" fate.env | awk -F '=' '{print $2}'`
package_dir_name="FATE_install_"${version}
package_dir=${source_dir}/cluster-deploy/${package_dir_name}
echo "[INFO] build info"
echo "[INFO] version: "${version}
echo "[INFO] version tag: "${version_tag}
echo "[INFO] package output dir is "${package_dir}
rm -rf ${package_dir} ${package_dir}_${version_tag}".tar.gz"
mkdir -p ${package_dir}

function packaging_bin() {
    echo "[INFO] package bin start"
    cp -r bin ${package_dir}/
    echo "[INFO] package bin done"
}

function packaging_conf() {
    echo "[INFO] package conf start"
    cp fate.env RELEASE.md python/requirements.txt ${package_dir}/
    cp -r conf ${package_dir}/
    echo "[INFO] package bin done"
}

function packaging_python(){
    echo "[INFO] package python start"
    if [[ ! -d "python/component_plugins/fate" ]];then
      pull_fate
    fi
    cp -r python ${package_dir}/
    echo "[INFO] package python done"
}

function packaging_examples(){
    echo "[INFO] package example start"
    cp -r examples ${package_dir}/
    echo "[INFO] package example done"
}

packaging_fateboard(){
    echo "[INFO] package fateboard start"
    cd ${source_dir}
    if [[ ! -d "fateboard" ]];then
      pull_fateboard
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
    echo "[INFO] package fateboard done"
}

pull_fate(){
    echo "[INFO] pull fate code start"
    cd ${source_dir}
    fate_git_url=`grep -A 3 '"fate"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
    fate_git_branch=`grep -A 3 '"fate"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
    echo "[INFO] git clone fate submodule source code from ${fate_git_url} branch ${fate_git_branch}"
    cd python/component_plugins/
    if [[ -d "fate" ]];then
        while [[ true ]];do
            read -p "The fate directory already exists, delete and re-download? [y/n] " input
            case ${input} in
            [yY]*)
                    echo "[INFO] delete the original fate"
                    rm -rf fateboard
                    git clone ${fate_git_url} -b ${fate_git_branch} --depth=1 fate
                    break
                    ;;
            [nN]*)
                    echo "[INFO] use the original fate"
                    break
                    ;;
            *)
                    echo "just enter y or n, please."
                    ;;
            esac
        done
    else
        git clone ${fate_git_url} -b ${fate_git_branch} --depth=1 fate
    fi
    echo "[INFO] pull fate code done"
    cd ${source_dir}
}

pull_fateboard(){
    echo "[INFO] get fateboard code start"
    cd ${source_dir}
    fateboard_git_url=`grep -A 3 '"fateboard"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
    fateboard_git_branch=`grep -A 3 '"fateboard"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
    echo "[INFO] git clone fateboard submodule source code from ${fateboard_git_url} branch ${fateboard_git_branch}"
    if [[ -d "fateboard" ]];then
        while [[ true ]];do
            read -p "The fateboard directory already exists, delete and re-download? [y/n] " input
            case ${input} in
            [yY]*)
                    echo "[INFO] delete the original fateboard"
                    rm -rf fateboard
                    git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
                    break
                    ;;
            [nN]*)
                    echo "[INFO] use the original fateboard"
                    break
                    ;;
            *)
                    echo "just enter y or n, please."
                    ;;
            esac
        done
    else
        git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
    fi
    echo "[INFO] get fateboard code done"
}

packaging_eggroll(){
    echo "[INFO] package eggroll start"
    cd ${source_dir}
    if [[ ! -d "eggroll" ]];then
      pull_eggroll
    fi
    cd ./eggroll
    cd ./deploy
    sh ./auto-packaging.sh
    mkdir -p ${package_dir}/eggroll
    mv ${source_dir}/eggroll/eggroll.tar.gz ${package_dir}/eggroll/
    cd ${package_dir}/eggroll/
    tar xzf eggroll.tar.gz
    rm -rf eggroll.tar.gz
    echo "[INFO] package eggroll done"
}

pull_eggroll(){
    echo "[INFO] get eggroll code start"
    cd ${source_dir}
    eggroll_git_url=`grep -A 3 '"eggroll"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
    eggroll_git_branch=`grep -A 3 '"eggroll"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
    echo "[INFO] git clone eggroll submodule source code from ${eggroll_git_url} branch ${eggroll_git_branch}"
    if [[ -d "eggroll" ]];then
        while [[ true ]];do
            read -p "the eggroll directory already exists, delete and re-download? [y/n] " input
            case ${input} in
            [yY]*)
                    echo "[INFO] delete the original eggroll"
                    rm -rf eggroll
                    git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
                    break
                    ;;
            [nN]*)
                    echo "[INFO] use the original eggroll"
                    break
                    ;;
            *)
                    echo "just enter y or n, please."
                    ;;
            esac
        done
    else
        git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
    fi
    echo "[INFO] get eggroll code done"
}

function packaging_proxy(){
    echo "[INFO] package proxy start"
    cd ${source_dir}
    cd c/proxy
    mkdir -p ${package_dir}/proxy/nginx
    cp -r conf lua ${package_dir}/proxy/nginx
    echo "[INFO] package proxy done"
}

compress(){
    echo "[INFO] compress start"
    cd ${package_dir}
    touch ./packages_md5.txt
    os_kernel=`uname -s`
    for module in "${packaging_modules[@]}";
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
    echo "[INFO] compress done"
    echo "[INFO] a total of `ls ${package_dir} | wc -l | awk '{print $1}'` packages:"
    ls -lrt ${package_dir}
    cd ${source_dir}/cluster-deploy/
    tar czf ${package_dir_name}_${version_tag}".tar.gz" ${package_dir_name}
}


build() {
    echo "[INFO] packaging start------------------------------------------------------------------------"
    for module in "${packaging_modules[@]}";
    do
        packaging_${module}
        echo
    done
    echo "[INFO] packaging end ------------------------------------------------------------------------"
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