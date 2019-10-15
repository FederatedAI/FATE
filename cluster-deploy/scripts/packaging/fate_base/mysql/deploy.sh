#!/bin/bash
set -e
module_name="mysql"
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./configurations.sh

config_path=$2
if [[ ${config_path} == "" ]] || [[ ! -f ${config_path} ]]
then
	echo "usage: $0 {install} {configurations path}."
	exit
fi
source ${config_path}


package_source(){
    cd ${output_packages_dir}/source
	if [[ -e "${module_name}" ]]
	then
		rm ${module_name}
	fi
	mkdir -p ${module_name}
	cd ${module_name}
	cp ${source_code_dir}/cluster-deploy/scripts/fate-base/packages/mysql-${mysql_version}-linux-glibc2.12-x86_64.tar.xz ./
	tar xf mysql-8.0.13-linux-glibc2.12-x86_64.tar.xz
	mv mysql-${mysql_version}-* mysql-${mysql_version}
	cd mysql-${mysql_version}
    mkdir data conf log
	return 0
}

config(){
    node_label=$3
	cd ${output_packages_dir}/config/${node_label}
	if [[ -e "${module_name}" ]]
	then
		rm ${module_name}
	fi
	mkdir -p ./${module_name}/conf
	cp -r ${cwd}/conf conf/
	cp ${cwd}/service.sh conf/

	cd ${module_name}
    cp ${cwd}/install.sh ./
    cp ${cwd}/${config_path} ./configurations.sh
    return 0
}

install() {
    mkdir -p ${deploy_dir}
    cp -r ${deploy_packages_dir}/source/${module_name} ${deploy_dir}/
    cp -r ${deploy_packages_dir}/config/${module_name}/conf/* ${deploy_dir}/
}

init(){
    mysql_dir=${deploy_dir}/${module_name}/mysql-${mysql_version}
    cd ${mysql_dir}
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

case "$1" in
    package_source)
        package_source $*
        ;;
    config)
        config $*
        ;;
    init)
        init $*
        ;;
    install)
        install $*
        ;;
	*)
		echo "usage: $0 {source_build|build|config|init|install} {configurations path}."
        exit -1
esac
