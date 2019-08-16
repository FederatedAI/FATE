#!/bin/bash
base="$(cd `dirname $0`; pwd)"
cdir="/data/projects/common/mysql"
mkdir -p $cdir
cd $cdir
tar -xvf $base/packages/mysql-8.0.13-linux-glibc2.12-x86_64.tar.xz
mv mysql-8.0.13-* mysql-8.0.13
cd mysql-8.0.13
mkdir data conf log
cp $base/conf/my.cnf ./conf/my.cnf
./bin/mysqld --initialize --user=app --basedir=$cdir/mysql-8.0.13  --datadir=$cdir/mysql-8.0.13/data &> install_init.log
tempstr=`cat install_init.log  | grep root@localhost`
passwdstr=${tempstr##* }
nohup ./bin/mysqld_safe --defaults-file=$cdir/mysql-8.0.13/conf/my.cnf --user=app &
sleep 60
$cdir/mysql-8.0.13/bin/mysql -uroot -p$passwdstr -S $cdir/mysql-8.0.13/mysql.sock --connect-expired-password << EOF
alter user  'root'@'localhost' IDENTIFIED by "fate_dev";
EOF
echo "the password of root: fate_dev"
