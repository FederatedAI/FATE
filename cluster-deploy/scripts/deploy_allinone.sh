#!/bin/bash
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source_dir=$(cd `dirname ${cwd}`; cd ../; pwd)
packaging_dir=${cwd}/packaging
packages_dir=${packaging_dir}/packages

modules=(metaservice egg proxy roll mysql)

source ./allinone-configurations.sh
mkdir -p ${packages_dir}

test() {
    echo "[INFO] check configuration"
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
}

usage() {
    echo "usage: $0 {all|[module1, ...]}"
}

fate_flow() {
    cp configurations.sh configurations.sh.tmp
    sed -i "s#source_dir=.*#source_dir=${source_dir}#g" ./configurations.sh.tmp
    sed -i "s#packages_dir=.*#packages_dir=${packages_dir}#g" ./configurations.sh.tmp
    sh install.sh package_source ./configurations.sh.tmp
    # sed -i "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./configurations.sh.tmp
    # sed -i "s#venv_dir=.*#venv_dir=${deploy_dir}/venv#g" ./configurations.sh.tmp
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
