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
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./default_configurations.sh
source ./allinone_cluster_configurations.sh

deploy_modes=(binary build)
support_modules=(jdk python mysql redis fate_flow federatedml fateboard proxy federation roll meta-service egg)
base_modules=(jdk python  mysql redis)
deploy_modules=()
eggroll_modules=(roll meta-service egg)
deploy_mode=$1
if_one_machine=1
source_code_dir=$(cd `dirname ${cwd}`; cd ../; pwd)
module_deploy_script_dir=${cwd}/deploy
output_packages_dir=$(cd `dirname ${cwd}`;pwd)/output_packages
deploy_packages_dir=${deploy_dir}/packages
mkdir -p ${output_packages_dir}

echo "[INFO] Check..."
if [[ ${deploy_modes[@]/${deploy_mode}/} != ${deploy_modes[@]} ]];then
    rm -rf ${output_packages_dir}/*
    mkdir -p ${output_packages_dir}/source
    mkdir -p ${output_packages_dir}/config
    for node_ip in "${node_list[@]}"; do
        mkdir -p ${output_packages_dir}/config/${node_ip}
    done
    if [[ ${#node_list[*]} -eq 1 ]];then
        if_one_machine=0
    fi
else
    echo "[INFO] can not support this deploy mode ${deploy_mode}"
    exit 1
fi

init_env() {
    for node_ip in "${node_list[@]}"; do
        ssh -tt ${user}@${node_ip} << eeooff
mkdir -p ${deploy_packages_dir}
exit
eeooff
        scp ${cwd}/deploy/fate_base/env.sh ${user}@${node_ip}:${deploy_packages_dir}
        ssh -tt ${user}@${node_ip} << eeooff
cd ${deploy_packages_dir}
sh env.sh
exit
eeooff
    done
}

if_base() {
    module_name=$1
    if [[ ${base_modules[@]/${module_name}/} != ${base_modules[@]} ]];then
        return 0
    else
        return 1
    fi
}

if_eggroll() {
    module_name=$1
    if [[ ${eggroll_modules[@]/${module_name}/} != ${eggroll_modules[@]} ]];then
        return 0
    else
        return 1
    fi
}

config_enter() {
    config_label=$1
    module_name=$2
	if [[ -e ${output_packages_dir}/config/${config_label}/${module_name} ]]
	then
		rm -rf ${output_packages_dir}/config/${config_label}/${module_name}
	fi
	mkdir -p ${output_packages_dir}/config/${config_label}/${module_name}/conf
    cp ./deploy.sh ${output_packages_dir}/config/${config_label}/${module_name}/
    cp ./configurations.sh.tmp ${output_packages_dir}/config/${config_label}/${module_name}/configurations.sh
}

packaging_jdk() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s/jdk_version=.*/jdk_version=${jdk_version}/g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_jdk() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    config_enter ${config_label} jdk
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_python() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s/python_version=.*/python_version=${python_version}/g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_python() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    config_enter ${config_label} python
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_mysql() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s/mysql_version=.*/mysql_version=${mysql_version}/g" ./configurations.sh.tmp
    sed -i.bak "s/user=.*/user=${user}/g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/mysql_user=.*/mysql_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i.bak "s/mysql_password=.*/mysql_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i.bak "s/fate_flow_db_name=.*/fate_flow_db_name=${fate_flow_db_name}/g" ./configurations.sh.tmp
    sed -i.bak "s/eggroll_meta_service_db_name=.*/eggroll_meta_service_db_name=${eggroll_meta_service_db_name}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_mysql() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i.bak "s/mysql_ip=.*/mysql_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/roll_ip=.*/roll_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/egg_ip=.*/egg_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/storage_service_ip=.*/storage_service_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/party_ips=.*/party_ips=\(${node_ip[*]}\)/g" ./configurations.sh.tmp
    config_enter ${config_label} mysql
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_redis() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s/redis_version=.*/redis_version=${redis_version}/g" ./configurations.sh.tmp
    sed -i.bak "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_redis() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    config_enter ${config_label} redis
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_fate_flow() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp

    sed -i.bak "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i.bak "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i.bak "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_fate_flow() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/python#g" ./configurations.sh.tmp
    sed -i.bak "s#python_path=.*#python_path=${party_deploy_dir}/python:${party_deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
    sed -i.bak "s#venv_dir=.*#venv_dir=${party_deploy_dir}/common/python/venv#g" ./configurations.sh.tmp
    sed -i.bak "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/db_name=.*/db_name=${fate_flow_db_name}/g" ./configurations.sh.tmp
    sed -i.bak "s/redis_ip=.*/redis_ip=${node_ip}/g" ./configurations.sh.tmp
	config_enter ${config_label} fate_flow
	sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_federatedml() {
    cp configurations.sh configurations.sh.tmp
	cp services.env service.env.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_federatedml() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/python#g" ./configurations.sh.tmp
    sed -i.bak "s#python_path=.*#python_path=${party_deploy_dir}/python:${party_deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
    sed -i.bak "s#venv_dir=.*#venv_dir=${party_deploy_dir}/common/python/venv#g" ./configurations.sh.tmp
    sed -i.bak "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s/roll.host=.*/roll.host=${node_ip}/g" ./service.env.tmp
    sed -i.bak "s/federation.host=.*/federation.host=${node_ip}/g" ./service.env.tmp
    sed -i.bak "s/fateflow.host=.*/fateflow.host=${node_ip}/g" ./service.env.tmp
    sed -i.bak "s/fateboard.host=.*/fateboard.host=${node_ip}/g" ./service.env.tmp
    sed -i.bak "s/proxy.host=.*/proxy.host=${node_ip}/g" ./service.env.tmp
    config_enter ${config_label} federatedml
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_fateboard() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s#version=.*#version=${version}#g" ./configurations.sh.tmp
    sed -i.bak "s#fateboard_version=.*#fateboard_version=${fateboard_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i.bak "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i.bak "s/node_list=.*/node_list=()/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_fateboard() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/fate_flow_ip=.*/fate_flow_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${config_label} fateboard
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_federation() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s#version=.*#version=${version}#g" ./configurations.sh.tmp
    sed -i.bak "s#federation_version=.*#federation_version=${federation_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_federation() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${config_label} federation
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_proxy() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s#version=.*#version=${version}#g" ./configurations.sh.tmp
    sed -i.bak "s#proxy_version=.*#proxy_version=${proxy_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_proxy() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    exchange_ip=${node_list[1-party_index]}
    sed -i.bak "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/federation_ip=.*/federation_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/fate_flow_ip=.*/fate_flow_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/exchange_ip=.*/exchange_ip=${exchange_ip}/g" ./configurations.sh.tmp
    config_enter ${config_label} proxy
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_roll() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s#version=.*#version=${version}#g" ./configurations.sh.tmp
    sed -i.bak "s#roll_version=.*#roll_version=${roll_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_roll() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${config_label} roll
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_metaservice() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s#version=.*#version=${version}#g" ./configurations.sh.tmp
    sed -i.bak "s#meta_service_version=.*#meta_service_version=${meta_service_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i.bak "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i.bak "s/db_name=.*/db_name=${eggroll_meta_service_db_name}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_metaservice() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${config_label} meta-service
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}

packaging_egg() {
    cp configurations.sh configurations.sh.tmp
    sed -i.bak "s#version=.*#version=${version}#g" ./configurations.sh.tmp
    sed -i.bak "s#egg_version=.*#egg_version=${egg_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_egg() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    config_label=$2
    party_deploy_dir=$3
    sed -i.bak "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i.bak "s#venv_dir=.*#venv_dir=${party_deploy_dir}/common/python/venv#g" ./configurations.sh.tmp
    sed -i.bak "s#python_path=.*#python_path=${party_deploy_dir}/python:${party_deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
    sed -i.bak "s#data_dir=.*#data_dir=${party_deploy_dir}/eggroll/data-dir#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/roll_ip=.*/roll_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${config_label} egg
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${config_label}
}


distribute() {
    cd ${output_packages_dir}
    echo "[INFO] compressed source"
    tar czf source.tar.gz ./source
    echo "[INFO] compressed source done"
    deploy_packages_dir=${deploy_dir}/packages
	for node_ip in "${node_list[@]}"; do
	    echo "[INFO] distribute source to ${node_ip}"
	    ssh -tt ${user}@${node_ip} << eeooff
rm -rf ${deploy_packages_dir}
mkdir -p ${deploy_packages_dir}/config
exit
eeooff
	    scp ${output_packages_dir}/source.tar.gz ${user}@${node_ip}:${deploy_packages_dir}
	    cd ${output_packages_dir}/config/${node_ip}
	    tar czf config.tar.gz ./*
	    scp config.tar.gz  ${user}@${node_ip}:${deploy_packages_dir}
	    echo "[INFO] distribute source to ${node_ip} done"
	done
}

install() {
	for node_ip in "${node_list[@]}"; do
	    echo "[INFO] install on ${node_ip}"
	    ssh -tt ${user}@${node_ip} << eeooff
cd ${deploy_packages_dir}
tar xzf source.tar.gz
tar xzf config.tar.gz -C config
exit
eeooff
        for module in "${deploy_modules[@]}"; do
            echo "[INFO] -----------------------------------------------"
	        echo "[INFO] Install ${module} on ${node_ip}"
            if_base ${module}
            if [[ $? -eq 0 ]];then
                module_deploy_dir=${deploy_dir}/common/${module}
            else
                if_eggroll ${module}
                if [[ $? -eq 0 ]];then
                    module_deploy_dir=${deploy_dir}/eggroll/${module}
                else
                    module_deploy_dir=${deploy_dir}/${module}
                fi
            fi
            echo "[INFO] ${module} deploy dir is ${module_deploy_dir}"
	        ssh -tt ${user}@${node_ip} << eeooff
	            rm -rf ${module_deploy_dir}
                cd ${deploy_packages_dir}/config/${module}
                sh ./deploy.sh ${deploy_mode} install ./configurations.sh
                sh ./deploy.sh ${deploy_mode} init ./configurations.sh
                exit
eeooff
	        echo "[INFO] Install ${module} on ${node_ip} done"
            echo "[INFO] -----------------------------------------------"
        done
	    echo "[INFO] Install on ${node_ip} done"
	done
}

packaging_module() {
    module=$1
    echo "[INFO] ${module} is packaging:"
    cd ${cwd}
    cd ${module_deploy_script_dir}
    if_base ${module}
    if [[ $? -eq 0 ]];then
        echo "[INFO] ${module} is base module"
        cd fate_base
    else
        if_eggroll ${module}
        if [[ $? -eq 0 ]];then
            echo "[INFO] ${module} is eggroll module"
            cd eggroll
        else
            echo "[INFO] ${module} is application module"
        fi
    fi
    cd ${module}

    if [[ "${module}" == "meta-service" ]]; then
        packaging_metaservice
    else
        packaging_${module}
    fi

    for ((i=0;i<${#node_list[*]};i++))
    do
        node_ip=${node_list[i]}
        party_id=${party_list[i]}
        if [[ "${module}" == "meta-service" ]]; then
            config_metaservice ${i} ${node_ip} ${deploy_dir}
        else
            config_${module} ${i} ${node_ip} ${deploy_dir}
        fi
    done

    if [[ $? -ne 0 ]];then
        echo "[INFO] ${module} packaging error."
        exit 255
    else
        echo "[INFO] ${module} packaging successfully."
    fi
}

deploy() {
    echo "[INFO] Packaging start------------------------------------------------------------------------"
    for module in ${deploy_modules[@]};
    do
        packaging_module ${module}
        echo
    done
    echo "[INFO] Packaging end ------------------------------------------------------------------------"

    echo "[INFO] Distribute start------------------------------------------------------------------------"
    distribute
    echo "[INFO] Distribute end------------------------------------------------------------------------"

    echo "[INFO] Install start ------------------------------------------------------------------------"
    install
    echo "[INFO] Install end ------------------------------------------------------------------------"
}

all() {
    init_env
    for ((i=0;i<${#support_modules[*]};i++))
    do
        deploy_modules[i]=${support_modules[i]}
	done
    deploy
}


multiple() {
    total=$#
    init_env
    for ((i=2;i<total+1;i++)); do
        deploy_modules[i]=${!i//\//}
    done
    deploy
}

usage() {
    echo "usage: $0 {binary|build} {all|[module1, ...]}"
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
