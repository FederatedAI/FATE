#!/bin/bash
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source_code_dir=$(cd `dirname ${cwd}`; cd ../; pwd)
packaging_dir=${cwd}/packaging
output_packages_dir=${packaging_dir}/packages

modules=(fate_flow)

source ./allinone_configurations.sh
deploy_packages_dir=${deploy_dir}/packages
mkdir -p ${output_packages_dir}

init() {
    echo "[INFO] check configuration"
    rm -rf ${output_packages_dir}/*
    mkdir -p ${output_packages_dir}/source
    mkdir -p ${output_packages_dir}/config
	for node_ip in "${nodes_ip[@]}"; do
	    mkdir -p ${output_packages_dir}/config/${node_ip}
	done
}

init

fate_flow() {
    # source install / apt install
    cp configurations.sh configurations.sh.tmp
    sed -i "s#source_code_dir=.*#source_code_dir=${source_code_dir}#g" ./configurations.sh.tmp
    sed -i "s#output_packages_dir=.*#output_packages_dir=${output_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    sed -i "s#deploy_packages_dir=.*#deploy_packages_dir=${deploy_packages_dir}#g" ./configurations.sh.tmp
    sed -i "s#venv_dir=.*#venv_dir=${deploy_dir}/venv#g" ./configurations.sh.tmp
    sed -i "s/db_user=.*/db_user=${db_auth[0]}/g" ./configurations.sh.tmp
    sed -i "s/db_password=.*/db_password=${db_auth[1]}/g" ./configurations.sh.tmp
    sed -i "s/redis_password=.*/redis_password=${redis_password}/g" ./configurations.sh.tmp
    sh install.sh package_source ./configurations.sh.tmp
	for node_ip in "${nodes_ip[@]}"; do
        sed -i "s/db_ip=.*/db_ip=${node_ip}/g" ./configurations.sh.tmp
        sed -i "s/redis_ip=.*/redis_ip=${node_ip}/g" ./configurations.sh.tmp
	    sh install.sh config ./configurations.sh.tmp ${node_ip}
	done
}

distribute() {
    cd ${output_packages_dir}
    tar czf source.tar.gz ./source
    echo "[INFO] distribute source and config"
    deploy_packages_dir=${deploy_dir}/packages
	for node_ip in "${nodes_ip[@]}"; do
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
	for node_ip in "${nodes_ip[@]}"; do
	    ssh -tt ${user}@${node_ip} << eeooff
cd ${deploy_packages_dir}
tar xzf source.tar.gz
cd source
cd ${deploy_packages_dir}
tar xzf config.tar.gz -C config
cd ${deploy_packages_dir}/config
for module in "${modules[@]}"; do
    cd ${module}
    ls -lrt
    sh ./install.sh install ./configurations.sh
    sh ./install.sh init
done
exit
eeooff
	done
}

all() {
	for module in "${modules[@]}"; do
        echo
		echo "[INFO] ${module} is deploying:"
        echo "=================================="
        cd ${packaging_dir}/${module}/
        ${module}
        echo "----------------------------------"
		echo "[INFO] ${module} is deployed over."
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
    echo "usage: $0 {all|[module1, ...]}"
}

case "$1" in
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
