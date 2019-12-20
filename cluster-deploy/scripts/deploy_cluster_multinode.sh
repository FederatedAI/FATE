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
source ./multinode_cluster_configurations.sh

deploy_modes=(binary build)
support_modules=(jdk python mysql redis fate_flow federatedml fateboard proxy federation roll meta-service egg)
base_modules=(jdk python mysql redis)
eggroll_modules=(roll meta-service egg)
env_modules=(jdk python)
deploy_modules=()
deploy_mode=$1
all_node_ips=()
a_ips=()
b_ips=()
a_jdk=()
b_jdk=()
a_python=()
b_python=()
source_code_dir=$(cd `dirname ${cwd}`; cd ../; pwd)
module_deploy_script_dir=${cwd}/deploy
output_packages_dir=$(cd `dirname ${cwd}`;pwd)/output_packages
deploy_packages_dir=${deploy_dir}/packages
mkdir -p ${output_packages_dir}

echo "[INFO] Check..."

get_all_node_ip() {
    for ((i=0;i<${#party_list[*]};i++))
    do
        party_name=${party_names[i]}
        for ((j=0;j<${#support_modules[*]};j++))
        do
            module=${support_modules[j]}
            if [[ "${module}" == "meta-service" ]]; then
                eval tmp_ips=\${${party_name}_metaservice}
            else
                eval tmp_ips=\${${party_name}_${module}[*]}
            fi
            for tmp_ip in ${tmp_ips[@]}
            do
                all_node_ips[${#all_node_ips[*]}]=${tmp_ip}
                if [[ "${party_name}" == "a" ]];then
                    a_ips[${#a_ips[*]}]=${tmp_ip}
                elif [[ "${party_name}" == "b" ]];then
                    b_ips[${#b_ips[*]}]=${tmp_ip}
                fi
            done
        done
	done
    all_node_ips=($(echo ${all_node_ips[*]} | sed 's/ /\n/g'|sort | uniq))
    a_ips=($(echo ${a_ips[*]} | sed 's/ /\n/g'|sort | uniq))
    b_ips=($(echo ${b_ips[*]} | sed 's/ /\n/g'|sort | uniq))
	#len=${#all_node_ips[*]}
	#for ((i=0;i<$len;i++))
	#do
	#    for ((j=$len-1;j>i;j--))
	#    do
	#        if [[ ${all_node_ips[i]} = ${all_node_ips[j]} ]];then
	#            unset all_node_ips[i]
	#        fi
	#    done
	#done
	#TODO: not all node need to deploy all env
    a_jdk=("${a_ips[@]}")
    b_jdk=("${b_ips[@]}")
    a_python=("${a_ips[@]}")
    b_python=("${b_ips[@]}")
}

if [[ ${deploy_modes[@]/${deploy_mode}/} != ${deploy_modes[@]} ]];then
    rm -rf ${output_packages_dir}/*
    mkdir -p ${output_packages_dir}/source
    mkdir -p ${output_packages_dir}/config
    get_all_node_ip
    echo "[INFO] Deploy on ${#all_node_ips[*]} node"
    for node_ip in ${all_node_ips[*]}; do
        mkdir -p ${output_packages_dir}/config/${node_ip}
    done
else
    echo "[INFO] can not support this deploy mode ${deploy_mode}"
    exit 1
fi

init_env() {
    for node_ip in ${all_node_ips[*]}; do
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

if_env() {
    module_name=$1
    if [[ ${env_modules[@]/${module_name}/} != ${env_modules[@]} ]];then
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
    for node_ip in ${all_node_ips[*]}
    do
        sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
        config_enter ${node_ip} jdk
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
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
    for node_ip in ${all_node_ips[*]}
    do
        sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
        config_enter ${node_ip} python
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    eval my_ip=\${${party_name}_mysql}

    eval db_ip=\${${party_name}_mysql}
    eval redis_ip=\${${party_name}_redis}
    eval roll_ip=\${${party_name}_roll}
    eval federation_ip=\${${party_name}_federation}
    eval proxy_ip=\${${party_name}_proxy}
    eval metaservice_ip=\${${party_name}_metaservice}
    eval egg_ips=\${${party_name}_egg[*]}
    if [[ "${party_name}" == "a" ]];then
        party_ips=("${a_ips[@]}")
    elif [[ "${party_name}" == "b" ]];then
        party_ips=("${b_ips[@]}")
    fi
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i.bak "s/mysql_ip=.*/mysql_ip=${db_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/proxy_ip=.*/proxy_ip=${proxy_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/roll_ip=.*/roll_ip=${roll_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/meta_service_ip=.*/meta_service_ip=${metaservice_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/egg_ip=.*/egg_ip=\(${egg_ips[*]}\)/g" ./configurations.sh.tmp
    sed -i.bak "s/storage_service_ip=.*/storage_service_ip=\(${egg_ips[*]}\)/g" ./configurations.sh.tmp
    sed -i.bak "s/party_ips=.*/party_ips=\(${party_ips[*]}\)/g" ./configurations.sh.tmp
    config_enter ${my_ip} mysql
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    eval my_ip=\${${party_name}_redis}
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    config_enter ${my_ip} redis
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    if [[ "${party_name}" == "a" ]];then
        my_ips=("${a_ips[@]}")
    elif [[ "${party_name}" == "b" ]];then
        my_ips=("${b_ips[@]}")
    fi

    eval db_ip=\${${party_name}_mysql}
    eval redis_ip=\${${party_name}_redis}
    for my_ip in ${my_ips[*]};do
        sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/python#g" ./configurations.sh.tmp
        sed -i.bak "s#python_path=.*#python_path=${deploy_dir}/python:${deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
        sed -i.bak "s#venv_dir=.*#venv_dir=${deploy_dir}/common/python/venv#g" ./configurations.sh.tmp
        sed -i.bak "s/db_ip=.*/db_ip=${db_ip}/g" ./configurations.sh.tmp
        sed -i.bak "s/db_name=.*/db_name=${fate_flow_db_name}/g" ./configurations.sh.tmp
        sed -i.bak "s/redis_ip=.*/redis_ip=${redis_ip}/g" ./configurations.sh.tmp
        config_enter ${my_ip} fate_flow
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
    done
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
    party_name=${party_names[party_index]}
    if [[ "${party_name}" == "a" ]];then
        my_ips=("${a_ips[@]}")
    elif [[ "${party_name}" == "b" ]];then
        my_ips=("${b_ips[@]}")
    fi

    eval roll_ip=\${${party_name}_roll}
    eval federation_ip=\${${party_name}_federation}
    eval fateflow_ip=\${${party_name}_fate_flow}
    eval fateboard_ip=\${${party_name}_fateboard}
    eval proxy_ip=\${${party_name}_proxy}
    for my_ip in ${my_ips[*]};do
        sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/python#g" ./configurations.sh.tmp
        sed -i.bak "s#python_path=.*#python_path=${deploy_dir}/python:${deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
        sed -i.bak "s#venv_dir=.*#venv_dir=${deploy_dir}/common/python/venv#g" ./configurations.sh.tmp
        sed -i.bak "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
        sed -i.bak "s/roll.host=.*/roll.host=${roll_ip}/g" ./service.env.tmp
        sed -i.bak "s/federation.host=.*/federation.host=${federation_ip}/g" ./service.env.tmp
        sed -i.bak "s/fateflow.host=.*/fateflow.host=${fateflow_ip}/g" ./service.env.tmp
        sed -i.bak "s/fateboard.host=.*/fateboard.host=${fateboard_ip}/g" ./service.env.tmp
        sed -i.bak "s/proxy.host=.*/proxy.host=${proxy_ip}/g" ./service.env.tmp
        config_enter ${my_ip} federatedml
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
    done
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
    party_name=${party_names[party_index]}
    eval my_ip=\${${party_name}_fateboard}

    eval db_ip=\${${party_name}_mysql}
    eval fateflow_ip=\${${party_name}_fate_flow}
    sed -i.bak "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/db_ip=.*/db_ip=${db_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/fate_flow_ip=.*/fate_flow_ip=${fateflow_ip}/g" ./configurations.sh.tmp
    config_enter ${my_ip} fateboard
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    eval my_ip=\${${party_name}_federation}

    eval db_ip=\${${party_name}_mysql}
    eval metaservice_ip=\${${party_name}_metaservice}
    eval proxy_ip=\${${party_name}_proxy}
    sed -i.bak "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/meta_service_ip=.*/meta_service_ip=${metaservice_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/proxy_ip=.*/proxy_ip=${proxy_ip}/g" ./configurations.sh.tmp
    config_enter ${my_ip} federation
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    eval my_ip=\${${party_name}_proxy}

    eval roll_ip=\${${party_name}_roll}
    eval federation_ip=\${${party_name}_federation}
    eval fateflow_ip=\${${party_name}_fate_flow}
    eval proxy_ip=\${${party_name}_proxy}
    eval exchange_ip=\${${party_names[1-party_index]}_proxy}
    sed -i.bak "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/proxy_ip=.*/proxy_ip=${proxy_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/federation_ip=.*/federation_ip=${federation_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/fate_flow_ip=.*/fate_flow_ip=${fateflow_ip}/g" ./configurations.sh.tmp
    sed -i.bak "s/exchange_ip=.*/exchange_ip=${exchange_ip}/g" ./configurations.sh.tmp
    config_enter ${my_ip} proxy
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    eval my_ip=\${${party_name}_roll}

    eval metaservice_ip=\${${party_name}_metaservice}
    sed -i.bak "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/meta_service_ip=.*/meta_service_ip=${metaservice_ip}/g" ./configurations.sh.tmp
    config_enter ${my_ip} roll
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    eval my_ip=\${${party_name}_metaservice}

    eval db_ip=\${${party_name}_mysql}
    sed -i.bak "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i.bak "s/db_ip=.*/db_ip=${db_ip}/g" ./configurations.sh.tmp
    config_enter ${my_ip} meta-service
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
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
    party_name=${party_names[party_index]}
    party_id=${party_list[${party_index}]}
    eval my_ips=\${${party_name}_egg[*]}
    eval roll_ip=\${${party_name}_roll}
    eval proxy_ip=\${${party_name}_proxy}
    eval clustercomm_ip=\${${party_name}_federation}
    for my_ip in ${my_ips[*]};do
        sed -i.bak "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-${jdk_version}#g" ./configurations.sh.tmp
        sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}/eggroll#g" ./configurations.sh.tmp
        sed -i.bak "s#venv_dir=.*#venv_dir=${deploy_dir}/common/python/venv#g" ./configurations.sh.tmp
        sed -i.bak "s#python_path=.*#python_path=${deploy_dir}/python:${deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
        sed -i.bak "s#data_dir=.*#data_dir=${deploy_dir}/eggroll/data-dir#g" ./configurations.sh.tmp
        sed -i.bak "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
        sed -i.bak "s/roll_ip=.*/roll_ip=${roll_ip}/g" ./configurations.sh.tmp
        sed -i.bak "s/proxy_ip=.*/proxy_ip=${proxy_ip}/g" ./configurations.sh.tmp
        sed -i.bak "s/clustercomm_ip=.*/clustercomm_ip=${clustercomm_ip}/g" ./configurations.sh.tmp
        config_enter ${my_ip} egg
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${my_ip}
    done
}


distribute() {
    cd ${output_packages_dir}
    echo "[INFO] Compressed source"
    tar czf source.tar.gz ./source
    echo "[INFO] Compressed source done"
    deploy_packages_dir=${deploy_dir}/packages
	for node_ip in "${all_node_ips[@]}"; do
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
	for node_ip in "${all_node_ips[@]}"; do
	    echo "[INFO] Decompressed on ${node_ip}"
	    ssh -tt ${user}@${node_ip} << eeooff
cd ${deploy_packages_dir}
tar xzf source.tar.gz
tar xzf config.tar.gz -C config
exit
eeooff
	    echo "[INFO] Decompressed on ${node_ip} done"
    done

    for ((i=0;i<${#party_list[*]};i++))
    do
        party_name=${party_names[i]}
        for module in "${deploy_modules[@]}"; do
	        echo "[INFO] Install ${module}"
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

            if_env ${module}
            if [[ $? -eq 0 ]];then
                #TODO: improve
                case ${party_name} in
                    "a")
                        case ${module} in
                            "jdk")
                                module_ips=("${a_jdk[*]}")
                            ;;
                            "python")
                                module_ips=("${a_python[*]}")
                            ;;
                        esac
                        ;;
                    "b")
                        case ${module} in
                            "jdk")
                                module_ips=("${b_jdk[*]}")
                            ;;
                            "python")
                                module_ips=("${b_python[*]}")
                            ;;
                        esac
                        ;;
                esac
            elif [[ "${module}" == "federatedml" ]];then
                module_ips=("${all_node_ips[*]}")
            elif [[ "${module}" == "fate_flow" ]];then
                module_ips=("${all_node_ips[*]}")
            elif [[ "${module}" == "meta-service" ]];then
                eval module_ips=\${${party_name}_metaservice[*]}
            else
                eval module_ips=\${${party_name}_${module}[*]}
            fi

            for node_ip in ${module_ips[*]}
            do
	            echo "[INFO] Install ${module} on ${node_ip}"
                echo "[INFO] -----------------------------------------------"
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
	        echo "[INFO] Install ${module} done"
        done
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

    for ((i=0;i<${#party_list[*]};i++))
    do
        if [[ "${module}" == "meta-service" ]]; then
            config_metaservice ${i}
        else
            config_${module} ${i}
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
