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
support_modules=(bin conf examples build deploy proxy fate fateflow fateboard eggroll)
environment_modules=(python36 jdk pypi)
packaging_modules=()
echo ${source_dir}
if [[ -n ${1} ]]; then
    version_tag=$1
else
    version_tag="rc"
fi

cd ${source_dir}
echo "[INFO] source dir: ${source_dir}"
git submodule init
git submodule update
version=`grep "FATE=" fate.env | awk -F '=' '{print $2}'`
package_dir_name="FATE_install_${version}_${version_tag}"
package_dir=${source_dir}/${package_dir_name}
echo "[INFO] build info"
echo "[INFO] version: "${version}
echo "[INFO] version tag: "${version_tag}
echo "[INFO] package output dir is "${package_dir}
rm -rf ${package_dir} ${package_dir}".tar.gz"
mkdir -p ${package_dir}

function packaging_bin() {
    packaging_general_dir "bin"
}

function packaging_conf() {
    echo "[INFO] package conf start"
    cp fate.env RELEASE.md python/requirements.txt ${package_dir}/
    cp -r conf ${package_dir}/
    echo "[INFO] package bin done"
}

function packaging_examples(){
    packaging_general_dir "examples"
}

function packaging_build(){
    packaging_general_dir "build"
}

function packaging_deploy(){
    packaging_general_dir "deploy"
}

function packaging_general_dir(){
    dir_name=$1
    echo "[INFO] package ${dir_name} start"
    if [[ -d "${package_dir}/${dir_name}" ]];then
      rm -rf ${package_dir}/${dir_name}
    fi
    cp -r ${dir_name} ${package_dir}/
    echo "[INFO] package ${dir_name} done"
}

function packaging_fate(){
    echo "[INFO] package fate start"
    if [[ -d "${package_dir}/fate" ]];then
      rm -rf ${package_dir}/fate
    fi
    mkdir -p ${package_dir}/fate
    cp -r python ${package_dir}/fate/
    echo "[INFO] package fate done"
}

function packaging_fateflow(){
    echo "[INFO] package fateflow start"
    #pull_fateflow
    cp -r fateflow ${package_dir}/
    echo "[INFO] package fateflow done"
}

packaging_fateboard(){
    echo "[INFO] package fateboard start"
    #pull_fateboard
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

packaging_eggroll(){
    echo "[INFO] package eggroll start"
    #pull_eggroll
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

pull_fateflow(){
    echo "[INFO] pull fateflow code start"
    cd ${source_dir}
    fateflow_git_url=`grep -A 3 '"fateflow"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
    fateflow_git_branch=`grep -A 3 '"fateflow"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
    echo "[INFO] git clone fateflow submodule source code from ${fateflow_git_url} branch ${fateflow_git_branch}"
    if [[ -d "fateflow" ]];then
        while [[ true ]];do
            read -p "The fateflow directory already exists, delete and re-download? [y/n] " input
            case ${input} in
            [yY]*)
                    echo "[INFO] delete the original fateflow"
                    rm -rf fateflow
                    git clone ${fateflow_git_url} -b ${fateflow_git_branch} --depth=1 fateflow
                    break
                    ;;
            [nN]*)
                    echo "[INFO] use the original fateflow"
                    break
                    ;;
            *)
                    echo "just enter y or n, please."
                    ;;
            esac
        done
    else
        git clone ${fateflow_git_url} -b ${fateflow_git_branch} --depth=1 fateflow
    fi
    echo "[INFO] pull fateflow code done"
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
    cd c/proxy
    mkdir -p ${package_dir}/proxy/nginx
    cp -r conf lua ${package_dir}/proxy/nginx/
    echo "[INFO] package proxy done"
}

function packaging_python36(){
    echo "[INFO] package python36 start"
    mkdir -p ${package_dir}/python36
    cd ${package_dir}/python36
    wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/Miniconda3-4.5.4-Linux-x86_64.sh
    echo "[INFO] package python36 done"
}

function packaging_jdk(){
    echo "[INFO] package jdk start"
    mkdir -p ${package_dir}/jdk
    cd ${package_dir}/jdk
    wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/jdk-8u192.tar.gz
    echo "[INFO] package jdk done"
}

function packaging_pypi(){
    echo "[INFO] package pypi start"
    mkdir -p ${package_dir}/pypi
    pip download -r ./python/requirements.txt -d ${package_dir}/pypi/
    echo "[INFO] package pypi done"
}

compress(){
    echo "[INFO] compress start"
    cd ${package_dir}
    touch ./packages_md5.txt
    os_kernel=`uname -s`
    find ./ -name ".*" | grep "DS_Store" | xargs -n1 rm -rf
    find ./ -name ".*" | grep "pytest_cache" | xargs -n1 rm -rf
    for module in "${packaging_modules[@]}";
    do
        case "${os_kernel}" in
            Darwin)
                gtar czf ${module}.tar.gz ./${module}
                md5_value=`md5 ${module}.tar.gz | awk '{print $4}'`
                ;;
            Linux)
                tar czf ${module}.tar.gz ./${module}
                md5_value=`md5sum ${module}.tar.gz | awk '{print $1}'`
                ;;
        esac
        echo "${module}:${md5_value}" >> ./packages_md5.txt
        rm -rf ./${module}
    done
    echo "[INFO] compress done"
    echo "[INFO] a total of `ls ${package_dir} | wc -l | awk '{print $1}'` packages:"
    ls -lrt ${package_dir}
    package_dir_parent=$(cd `dirname ${package_dir}`; pwd)
    cd ${package_dir_parent}
    tar czf ${package_dir_name}".tar.gz" ${package_dir_name}
}


build() {
    echo "[INFO] packaging start------------------------------------------------------------------------"
    for module in "${packaging_modules[@]}";
    do
        cd ${source_dir}
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