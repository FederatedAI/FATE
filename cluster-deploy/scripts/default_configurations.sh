#!/bin/bash
fate_cos_address=https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com
version=1.3.0
egg_version=
meta_service_version=
roll_version=
federation_version=
proxy_version=
fateboard_version=
python_version=1.3.0
jdk_version=8u192
mysql_version=8.0.13
redis_version=5.0.2
fate_flow_db_name=fate_flow
eggroll_meta_service_db_name=eggroll_meta

package_init() {
    output_packages_dir=$1
    module_name=$2
    cd ${output_packages_dir}/source
    if [[ -e "${module_name}" ]]
    then
        rm -rf ${module_name}
    fi
    mkdir -p ${module_name}
    cd ${module_name}
}

get_module_package() {
    source_code_dir=$1
    module_name=$2
    module_binary_package=$3
    echo "[INFO] Get ${module_name} package"
    copy_path=${source_code_dir}/cluster-deploy/packages/${module_binary_package}
    download_uri=${fate_cos_address}/${module_binary_package}
    if [[ -f ${copy_path} ]];then
        echo "[INFO] Copying ${copy_path}"
        cp ${copy_path} ./
    else
        echo "[INFO] Downloading ${download_uri}"
        wget -P ${source_code_dir}/cluster-deploy/packages/ ${download_uri}
        echo "[INFO] Finish downloading ${download_uri}"
        echo "[INFO] Copying ${copy_path}"
        cp ${copy_path} ./
    fi
    echo "[INFO] Finish get ${module_name} package"
}
