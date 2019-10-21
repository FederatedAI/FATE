#!/bin/bash
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./default_configurations.sh
source ./allinone_configurations.sh

deploy_modes=(binary build)
#support_modules=(jdk python mysql redis fate_flow federatedml fateboard proxy federation)
support_modules=(egg)
base_modules=(jdk python  mysql redis)
eggroll_modules=(roll meta-service egg)
deploy_mode=$1
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
    node_label=$1
    module_name=$2
	if [[ -e ${output_packages_dir}/config/${node_label}/${module_name} ]]
	then
		rm -rf ${output_packages_dir}/config/${node_label}/${module_name}
	fi
	mkdir -p ${output_packages_dir}/config/${node_label}/${module_name}/conf
    cp ./deploy.sh ${output_packages_dir}/config/${node_label}/${module_name}/
    cp ./configurations.sh.tmp ${output_packages_dir}/config/${node_label}/${module_name}/configurations.sh
}

deploy_jdk() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/jdk_version=.*/jdk_version=${jdk_version}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
	    config_enter ${node_ip} jdk
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}


deploy_python() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/python_version=.*/python_version=${python_version}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
	    config_enter ${node_ip} python
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}

deploy_mysql() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/mysql_version=.*/mysql_version=${mysql_version}/g" ./configurations.sh.tmp
    sed -i "s/user=.*/user=${user}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s/mysql_password=.*/mysql_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
	    config_enter ${node_ip} mysql
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}

deploy_redis() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/redis_version=.*/redis_version=${redis_version}/g" ./configurations.sh.tmp
    sed -i "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
	    config_enter ${node_ip} redis
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}

deploy_fate_flow() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/python#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp

    sed -i "s#python_path=.*#python_path=${deploy_dir}/python:${deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
    sed -i "s#venv_dir=.*#venv_dir=${deploy_dir}/common/python/miniconda3-fate-${python_version}#g" ./configurations.sh.tmp
    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
        sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/redis_ip=.*/redis_ip=${node_ip}/g" ./configurations.sh.tmp
	    config_enter ${node_ip} fate_flow
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}

deploy_federatedml() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/python#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
	    config_enter ${node_ip} federatedml
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}

deploy_fateboard() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/node_list=.*/node_list=()/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
    for node_ip in "${node_list[@]}"; do
        sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/fate_flow_ip=.*/fate_flow_ip=${node_ip}/g" ./configurations.sh.tmp
	    config_enter ${node_ip} fateboard
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
}

deploy_federation() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
    for ((i=0;i<${#node_list[*]};i++))
    do
        node_ip=${node_list[i]}
        party_id=${party_list[i]}
        sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
        sed -i "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
	    config_enter ${node_ip} federation
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
}

deploy_proxy() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
    for ((i=0;i<${#node_list[*]};i++))
    do
        node_ip=${node_list[i]}
        exchange_ip=${node_list[1-i]}
        party_id=${party_list[i]}
        sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
        sed -i "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/federation_ip=.*/federation_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/fate_flow_ip=.*/fate_flow_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/exchange_ip=.*/exchange_ip=${exchange_ip}/g" ./configurations.sh.tmp
	    config_enter ${node_ip} proxy
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
}

deploy_roll() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
    for ((i=0;i<${#node_list[*]};i++))
    do
        node_ip=${node_list[i]}
        party_id=${party_list[i]}
        sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
        sed -i "s/meta_service_ip=.*/meta_service_ip=${node_ip}/g" ./configurations.sh.tmp
	    config_enter ${node_ip} roll
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
}

deploy_metaservice() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp

    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp

    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
    for ((i=0;i<${#node_list[*]};i++))
    do
        node_ip=${node_list[i]}
        party_id=${party_list[i]}
        sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
        sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
	    config_enter ${node_ip} meta-service
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
}

deploy_egg() {
    cp configurations.sh configurations.sh.tmp
    sed -i"" "s#java_dir=.*#java_dir=${deploy_dir}/common/jdk/jdk-8u192#g" ./configurations.sh.tmp
    sed -i"" "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_dir=.*#deploy_dir=${deploy_dir}/eggroll#g" ./configurations.sh.tmp
    sed -i"" "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp

    sed -i "s#venv_dir=.*#venv_dir=${deploy_dir}/common/python/miniconda3-fate-${python_version}#g" ./configurations.sh.tmp
    sed -i "s#python_path=.*#python_path=${deploy_dir}/python:${deploy_dir}/eggroll/python#g" ./configurations.sh.tmp
    sed -i "s#data_dir=.*#data_dir=${deploy_dir}/eggroll/data_dir#g" ./configurations.sh.tmp

    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
    for ((i=0;i<${#node_list[*]};i++))
    do
        node_ip=${node_list[i]}
        party_id=${party_list[i]}
        sed -i "s/party_id=.*/party_id=${party_id}/g" ./configurations.sh.tmp
        sed -i "s/roll_ip=.*/roll_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/proxy_ip=.*/proxy_ip=${node_ip}/g" ./configurations.sh.tmp
	    config_enter ${node_ip} egg
        sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
    done
}


distribute() {
    cd ${output_packages_dir}
    tar czf source.tar.gz ./source
    echo "[INFO] distribute source and config"
    deploy_packages_dir=${deploy_dir}/packages
	for node_ip in "${node_list[@]}"; do
	    ssh -tt ${user}@${node_ip} << eeooff
rm -rf ${deploy_packages_dir}/*
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
        for module in "${support_modules[@]}"; do
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

all() {
    init_env
    echo "------------------------------------------------------------------------"
	for module in "${support_modules[@]}"; do
        echo
		echo "[INFO] ${module} is packaging:"
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
        if [[ $? -ne 0 ]];then
		    echo "[INFO] ${module} packaging error."
		    exit 255
		else
		    echo "[INFO] ${module} packaging successfully."
		fi
		cd ${cwd}
	done
    echo "------------------------------------------------------------------------"
	distribute
	install
}

multiple() {
    total=$#
    for (( i=1; i<total+1; i++)); do
        module=${!i//\//}
        echo
		echo "[INFO] ${module} is deploying:"
        echo "=================================="
        cd ${packaging_dir}/${module}/
        ${module}
        echo "-----------------------------------"
		echo "[INFO] ${module} is deployed over."
		cd ${cwd}
    done
    distribute
    install
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