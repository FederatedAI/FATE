#!/bin/bash
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./default_configurations.sh
source ./allinone_configurations.sh

deploy_modes=(binary build)
support_modules=(jdk python mysql redis fate_flow federatedml fateboard proxy federation roll meta-service egg)
base_modules=(jdk python  mysql redis)
deploy_modules=()
eggroll_modules=(roll meta-service egg)
deploy_mode=$1
if_one_machine=1
source_code_dir=$(cd `dirname ${cwd}`; cd ../; pwd)
packaging_dir=${cwd}/packaging
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
    if [[ "${deploy_mode}" == "binary" ]]; then
        # TODO: All modules support binary deployment mode and need to be removed here.
        for node_ip in "${node_list[@]}"; do
		    ssh -tt ${user}@${node_ip} << eeooff
mkdir -p ${deploy_packages_dir}
exit
eeooff
	        scp ${cwd}/packaging/fate_base/env.sh ${user}@${node_ip}:${deploy_packages_dir}
	        ssh -tt ${user}@${node_ip} << eeooff
cd ${deploy_packages_dir}
sh env.sh
exit
eeooff
        done
    elif [[ "${deploy_mode}" == "build" ]]; then
        echo "not support"
    fi
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
    party_label=$1
    module_name=$2
	if [[ -e ${output_packages_dir}/config/${party_label}/${module_name} ]]
	then
		rm -rf ${output_packages_dir}/config/${party_label}/${module_name}
	fi
	mkdir -p ${output_packages_dir}/config/${party_label}/${module_name}/conf
    cp ./deploy.sh ${output_packages_dir}/config/${party_label}/${module_name}/
    cp ./configurations.sh.tmp ${output_packages_dir}/config/${party_label}/${module_name}/configurations.sh
}

deploy_jdk() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/jdk_version=.*/jdk_version=${jdk_version}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_jdk() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    config_enter ${party_label} jdk
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_python() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/python_version=.*/python_version=${python_version}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_python() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    config_enter ${party_label} python
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_mysql() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/mysql_version=.*/mysql_version=${mysql_version}/g" ./configurations.sh.tmp
    sed -i "s/user=.*/user=${user}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s/mysql_password=.*/mysql_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/fate_flow_db_name=.*/fate_flow_db_name=${fate_flow_db_name}/g" ./configurations.sh.tmp
    sed -i "s/eggroll_meta_service_db_name=.*/eggroll_meta_service_db_name=${eggroll_meta_service_db_name}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_mysql() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i "s/mysql_ip=.*/mysql_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/roll_ip=.*/roll_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/egg_ip=.*/egg_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/storage_service_ip=.*/storage_service_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${party_label} mysql
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_redis() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/redis_version=.*/redis_version=${redis_version}/g" ./configurations.sh.tmp
    sed -i "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_redis() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/common#g" ./configurations.sh.tmp
    config_enter ${party_label} redis
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_fate_flow() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp

    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_fate_flow() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/python#g" ./configurations.sh.tmp
    sed -i "s#python_path=.*#python_path=${party_deploy_dir}/python:${party_deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
    sed -i "s#venv_dir=.*#venv_dir=${party_deploy_dir}/common/python/miniconda3-fate-${python_version}#g" ./configurations.sh.tmp
    sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/db_name=.*/db_name=${fate_flow_db_name}/g" ./configurations.sh.tmp
    sed -i "s/redis_ip=.*/redis_ip=${node_ip}/g" ./configurations.sh.tmp
	config_enter ${party_label} fate_flow
	sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_federatedml() {
    cp configurations.sh configurations.sh.tmp
	cp services.env service.env.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_federatedml() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/python#g" ./configurations.sh.tmp
    sed -i "s/roll.host=.*/roll.host=${node_ip}/g" ./service.env.tmp
    sed -i "s/federation.host=.*/federation.host=${node_ip}/g" ./service.env.tmp
    sed -i "s/fateflow.host=.*/fateflow.host=${node_ip}/g" ./service.env.tmp
    sed -i "s/fateboard.host=.*/fateboard.host=${node_ip}/g" ./service.env.tmp
    sed -i "s/proxy.host=.*/proxy.host=${node_ip}/g" ./service.env.tmp
    config_enter ${party_label} federatedml
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_fateboard() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/node_list=.*/node_list=()/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_fateboard() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i"" "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}#g" ./configurations.sh.tmp
    sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/fate_flow_ip=.*/fate_flow_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${party_label} fateboard
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_federation() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_federation() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i"" "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}#g" ./configurations.sh.tmp
    sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${party_label} federation
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_proxy() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_proxy() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    exchange_ip=${node_list[1-party_index]}
    sed -i"" "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}#g" ./configurations.sh.tmp
    sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/federation_ip=.*/federation_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/fate_flow_ip=.*/fate_flow_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/exchange_ip=.*/exchange_ip=${exchange_ip}/g" ./configurations.sh.tmp
    config_enter ${party_label} proxy
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_roll() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_roll() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i"" "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${party_label} roll
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_metaservice() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/db_name=.*/db_name=${eggroll_meta_service_db_name}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_metaservice() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i"" "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${party_label} meta-service
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}

deploy_egg() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} packaging ./configurations.sh.tmp
}

config_egg() {
    party_index=$1
    node_ip=${node_list[${party_index}]}
    party_id=${party_list[${party_index}]}
    party_label=$2
    party_deploy_dir=$3
    sed -i"" "s#java_dir=.*#java_dir=${party_deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${party_deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i "s#venv_dir=.*#venv_dir=${party_deploy_dir}/common/python/miniconda3-fate-${python_version}#g" ./configurations.sh.tmp
    sed -i "s#python_path=.*#python_path=${party_deploy_dir}/python:${party_deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
    sed -i "s#data_dir=.*#data_dir=${party_deploy_dir}/eggroll/data-dir#g" ./configurations.sh.tmp
    sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
    sed -i "s/roll_ip=.*/roll_ip=${node_ip}/g" ./configurations.sh.tmp
    sed -i "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
    config_enter ${party_label} egg
    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${party_label}
}


distribute() {
    cd ${output_packages_dir}
    tar czf source.tar.gz ./source
    echo "[INFO] distribute source and config"
    deploy_packages_dir=${deploy_dir}/packages
	for node_ip in "${node_list[@]}"; do
	    ssh -tt ${user}@${node_ip} << eeooff
rm -rf ${deploy_packages_dir}
mkdir -p ${deploy_packages_dir}/config
exit
eeooff
	    scp ${output_packages_dir}/source.tar.gz ${user}@${node_ip}:${deploy_packages_dir}
	    cd ${output_packages_dir}/config/${node_ip}
	    tar czf config.tar.gz ./*
	    scp config.tar.gz  ${user}@${node_ip}:${deploy_packages_dir}
	done
    echo "[INFO] distribute source and config done"
}

install() {
	for node_ip in "${node_list[@]}"; do
	    ssh -tt ${user}@${node_ip} << eeooff
cd ${deploy_packages_dir}
tar xzf source.tar.gz
tar xzf config.tar.gz -C config
exit
eeooff
        for module in "${deploy_modules[@]}"; do
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
        done
	done
}

deploy_module() {
    module=$1
    echo "[INFO] ${module} is packaging:"
    cd ${cwd}
    cd ${packaging_dir}
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
        deploy_metaservice
    else
        deploy_${module}
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
    echo "------------------------------------------------------------------------"
    for module in ${deploy_modules[@]};
    do
        deploy_module ${module}
        echo
    done
    echo "------------------------------------------------------------------------"
    distribute
    install
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