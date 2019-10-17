#!/bin/bash
set -e
module_name="mysql"
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./configurations.sh
source ../../../default_configurations.sh

usage() {
	echo "usage: $0 {apt/build} {package|config|install|init} {configurations path}."
}

deploy_mode=$1
config_path=$3
if [[ ${config_path} == "" ]] || [[ ! -f ${config_path} ]]
then
	usage
	exit
fi
source ${config_path}

package(){
    if [[ "${deploy_mode}" == "apt" ]]; then
        cd ${output_packages_dir}/source
        if [[ -e "${module_name}" ]]
        then
            rm ${module_name}
        fi
        mkdir -p ${module_name}
        cd ${module_name}
        wget ${fate_cos_address}/mysql-${mysql_version}-linux-glibc2.12-x86_64.tar.xz ./
        tar xf mysql-${mysql_version}-linux-glibc2.12-x86_64.tar.xz
        rm -rf mysql-${mysql_version}-linux-glibc2.12-x86_64.tar.xz
        mv mysql-${mysql_version}-linux-glibc2.12-x86_64 mysql-${mysql_version}
    elif [[ "${deploy_mode}" == "build" ]]; then
        echo "not support"
    fi
	return 0
}

config(){
    node_label=$4
	cd ${output_packages_dir}/config/${node_label}
	if [[ -e "${module_name}" ]]
	then
		rm ${module_name}
	fi
	mkdir -p ./${module_name}/conf
	cp -r ${cwd}/conf ./${module_name}/conf/
	cp ${cwd}/service.sh ./${module_name}/conf/

	cd ${module_name}
    cp ${cwd}/deploy.sh ./
    cp ${cwd}/${config_path} ./configurations.sh
    return 0
}

install() {
    mkdir -p ${deploy_dir}
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/${module_name}/mysql-${mysql_version}/
}

init(){
    mysql_dir=${deploy_dir}/${module_name}/mysql-${mysql_version}
    cd ${mysql_dir}
    mkdir data
    mkdir conf
    mkdir log
    ./bin/mysqld --initialize --user=${user} --basedir=${mysql_dir}  --datadir=${mysql_dir}/data &> install_init.log
    temp_str=`cat install_init.log  | grep root@localhost`
    password_str=${temp_str##* }
    nohup ./bin/mysqld_safe --defaults-file=${mysql_dir}/conf/my.cnf --user=${user} &
    sleep 10
    ${mysql_dir}/bin/mysql -uroot -p${password_str}-S ${mysql_dir}/mysql.sock --connect-expired-password << EOF
    alter user  'root'@'localhost' IDENTIFIED by "${mysql_password}";
EOF
    echo "the password of root: ${mysql_password}"
}

case "$2" in
    package)
        package $*
        ;;
    config)
        config $*
        ;;
    install)
        install $*
        ;;
    init)
        init $*
        ;;
	*)
	    usage
        exit -1
esac

