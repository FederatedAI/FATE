#!/bin/bash
base="$(cd `dirname $0`; pwd)"
cdir="/data/projects/common/mysql"
mkdir -p $cdir
cd $cdir
tar -xvf $base/packages/mysql-8.0*-linux*.tar.xz
mv mysql-8.0* mysql-8.0
cd mysql-8.0
mkdir data conf log
cp $base/conf/my.cnf ./conf/my.cnf
./bin/mysqld --initialize --user=app --basedir=$cdir/mysql-8.0  --datadir=$cdir/mysql-8.0/data
nohup ./bin/mysqld_safe --defaults-file=$cdir/mysql-8.0/conf/my.cnf --user=app &

