#!/bin/bash
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./default_configurations.sh
source ./allinone_configurations.sh

deploy_modes=(apt build)
#support_modules=(jdk python mysql redis fate_flow federatedml fateboard proxy federation)
support_modules=(mysql)
base_modules=(jdk python  mysql redis)

deploy_mode=$1
source_code_dir=$(cd `dirname ${cwd}`; cd ../; pwd)
packaging_dir=${cwd}/packaging
output_packages_dir=$(cd `dirname ${cwd}`;pwd)/packages
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
    if [[ "${deploy_mode}" == "apt" ]]; then
        # TODO: All modules support apt deployment mode and need to be removed here.
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


deploy_jdk() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s/jdk_version=.*/jdk_version=${jdk_version}/g" ./configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}


deploy_python() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/common#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
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
	    sh ./deploy.sh ${deploy_mode} config ./configurations.sh.tmp ${node_ip}
	done
}

deploy_fate_flow() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}/python#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#venv_dir=.*#venv_dir=${deploy_dir}/common/python#g" ./configurations.sh.tmp
    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sh ./deploy.sh ${deploy_mode} package ./configurations.sh.tmp
	for node_ip in "${node_list[@]}"; do
        sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/redis_ip=.*/redis_ip=${node_ip}/g" ./configurations.sh.tmp
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
            if [[ $(if_base ${module}) -eq 0 ]];then
                module_deploy_dir=${deploy_dir}/common/${module_name}
            else
                module_deploy_dir=${deploy_dir}/${module_name}
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
	for module in "${support_modules[@]}"; do
        echo
		echo "[INFO] ${module} is packaging:"
        echo "------------------------------------------------------------------------"
        cd ${packaging_dir}
        if [[ $(if_base ${module}) -eq 0 ]];then
            echo "[INFO] ${module} is base module"
            cd fate_base
        else
            echo "[INFO] ${module} is application module"
        fi
        cd ${module}
        deploy_${module}
        if [[ $? -ne 0 ]];then
		    echo "[INFO] ${module} packaging error."
		    exit 255
		else
		    echo "[INFO] ${module} packaging successfully."
		fi
        echo "------------------------------------------------------------------------"
		cd ${cwd}
	done
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
    echo "usage: $0 {apt|build} {all|[module1, ...]}"
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
