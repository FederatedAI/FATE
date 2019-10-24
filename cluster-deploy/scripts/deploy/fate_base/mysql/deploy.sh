#!/bin/bash
set -e
module_name="mysql"
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./configurations.sh

usage() {
	echo "usage: $0 {binary/build} {packaging|config|install|init} {configurations path}."
}

deploy_mode=$1
config_path=$3
if [[ ${config_path} == "" ]] || [[ ! -f ${config_path} ]]
then
	usage
	exit
fi
source ${config_path}

packaging(){
    source ../../../default_configurations.sh
    package_init ${output_packages_dir} ${module_name}
    if [[ "${deploy_mode}" == "binary" ]]; then
        get_module_binary ${source_code_dir} ${module_name} mysql-${mysql_version}-linux-glibc2.12-x86_64.tar.xz
        tar xf mysql-${mysql_version}-linux-glibc2.12-x86_64.tar.xz
        rm -rf mysql-${mysql_version}-linux-glibc2.12-x86_64.tar.xz
        mv mysql-${mysql_version}-linux-glibc2.12-x86_64 mysql-${mysql_version}
    elif [[ "${deploy_mode}" == "build" ]]; then
        echo "not support"
    fi
	return 0
}

config(){
    config_label=$4
	cd ${output_packages_dir}/config/${config_label}
	cp ${cwd}/service.sh ./${module_name}/conf/
	mkdir -p ./${module_name}/conf/conf
	cp -r ${cwd}/conf/* ./${module_name}/conf/conf/

    cd ./${module_name}/conf/
	cp ${source_code_dir}/eggroll/framework/meta-service/src/main/resources/create-meta-service.sql ./
	sed -i.bak "s/eggroll_meta/${eggroll_meta_service_db_name}/g" ./create-meta-service.sql

	echo > ./insert-node.sql
    echo "INSERT INTO node (ip, port, type, status) values ('${roll_ip}', '${roll_port}', 'ROLL', 'HEALTHY');" >> ./insert-node.sql
    echo "INSERT INTO node (ip, port, type, status) values ('${proxy_port}', '${proxy_port}', 'PROXY', 'HEALTHY');" >> ./insert-node.sql
    for ((i=0;i<${#egg_ip[*]};i++))
    do
        echo "INSERT INTO node (ip, port, type, status) values ('${egg_ip[i]}', '${egg_port}', 'EGG', 'HEALTHY');" >> ./insert-node.sql
    done
    for ((i=0;i<${#storage_service_ip[*]};i++))
    do
        echo "INSERT INTO node (ip, port, type, status) values ('${storage_service_ip[i]}', '${storage_service_port}', 'STORAGE', 'HEALTHY');" >> ./insert-node.sql
    done
    echo "show tables;" >> ./insert-node.sql
    echo "select * from node;" >> ./insert-node.sql

	cd ./conf
	sed -i.bak "s#basedir=.*#basedir=${deploy_dir}/${module_name}/mysql-${mysql_version}#g" ./my.cnf
	sed -i.bak "s#datadir=.*#datadir=${deploy_dir}/${module_name}/mysql-${mysql_version}/data#g" ./my.cnf
	sed -i.bak "s#socket=.*#socket=${deploy_dir}/${module_name}/mysql-${mysql_version}/mysql.sock#g" ./my.cnf
	sed -i.bak "s#log-error=.*#log-error=${deploy_dir}/${module_name}/mysql-${mysql_version}/log/mysqld.log#g" ./my.cnf
	sed -i.bak "s#pid-file=.*#pid-file=${deploy_dir}/${module_name}/mysql-${mysql_version}/data/mysqld.pid#g" ./my.cnf
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
    mkdir log
    sh service.sh stop
    ./bin/mysqld --initialize --user=${user} --basedir=${mysql_dir}  --datadir=${mysql_dir}/data &> install_init.log
    temp_str=`cat install_init.log  | grep root@localhost`
    password_str=${temp_str##* }
    nohup ./bin/mysqld_safe --defaults-file=${mysql_dir}/conf/my.cnf --user=${user} &
    sleep 10
    ./bin/mysql -uroot -p"${password_str}" -S ./mysql.sock --connect-expired-password << EOF
    ALTER USER 'root'@'localhost' IDENTIFIED by "${mysql_password}";
    CREATE USER 'root'@"${mysql_ip}" IDENTIFIED BY "${mysql_password}";
    GRANT ALL ON *.* TO 'root'@"${mysql_ip}";
    CREATE DATABASE ${fate_flow_db_name};
    source ${mysql_dir}/create-meta-service.sql;
    source ${mysql_dir}/insert-node.sql;
EOF
    echo "the password of root: ${mysql_password}"
}

case "$2" in
    packaging)
        packaging $*
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

