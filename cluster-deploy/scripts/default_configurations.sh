#!/bin/bash
fate_cos_address=https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com
python_version=1.1
jdk_version=8u192
mysql_version=8.0.13
redis_version=5.0.2

package_init() {
    output_packages_dir=$1
    module_name=$2
    cd ${output_packages_dir}/source
    if [[ -e "${module_name}" ]]
    then
        rm ${module_name}
    fi
    mkdir -p ${module_name}
    cd ${module_name}
}

get_module_binary() {
    source_code_dir=$1
    module_name=$2
    module_binary_package=$3
    echo "[INFO] Get ${module_name} available binary"
    copy_path=${source_code_dir}/cluster-deploy/packages/${module_binary_package}
    download_uri=${fate_cos_address}/${module_binary_package}
    if [[ -f ${copy_path} ]];then
        echo "[INFO] Copying ${copy_path}"
        cp ${copy_path} ./
    else
        echo "[INFO] Downloading ${download_uri}"
        wget ${download_uri}
    fi
}
