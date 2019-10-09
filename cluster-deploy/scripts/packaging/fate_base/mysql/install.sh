#!/bin/bash
set -e
if [[ $2 == "" ]]
then
echo "usage: $0 {install} {configurations path}."
exit
else
  source $2
fi
base="$(cd `dirname $0`; pwd)"
cdir=$mysqldir
cd $base
echo $base
cd ..
basedir=`pwd`
echo $basedir
if [ ! -d "${cdir}" ];then
    mkdir -p $cdir
fi
echo $cdir
cd $cdir
#source build
source_build() {
        return 0
}
#build
build(){
if [ ! -f "$basedir/packages/mysql-$mysqlversion-linux-glibc2.12-x86_64.tar.xz" ];then
    echo "No mysql-$mysqlversion-linux-glibc2.12-x86_64.tar.xz package can't be installed"
    exit
fi
tar -xvf $basedir/packages/mysql-$mysqlversion-linux-glibc2.12-x86_64.tar.xz
mv mysql-$mysqlversion-* mysql-$mysqlversion
cd $cdir/mysql-$mysqlversion
mkdir data conf log
}
#configuration
config(){
cd $cdir/mysql-$mysqlversion
cp $basedir/mysql/conf/my.cnf ./conf/my.cnf
}
#init service
init(){
cd $cdir/mysql-$mysqlversion
cp $basedir/mysql/service.sh $cdir/mysql-$mysqlversion
./bin/mysqld --initialize --user=$user --basedir=$cdir/mysql-$mysqlversion  --datadir=$cdir/mysql-$mysqlversion/data &> install_init.log
tempstr=`cat install_init.log  | grep root@localhost`
passwdstr=${tempstr##* }
nohup ./bin/mysqld_safe --defaults-file=$cdir/mysql-$mysqlversion/conf/my.cnf --user=$user &
sleep 10
$cdir/mysql-$mysqlversion/bin/mysql -uroot -p$passwdstr -S $cdir/mysql-$mysqlversion/mysql.sock --connect-expired-password << EOF
alter user  'root'@'localhost' IDENTIFIED by "$jdbcpass";
EOF
echo "the password of root: $jdbcpass"
}

case "$1" in
    install)
                source_build
                build
               config
               init
        ;;
        *)
                echo "usage: $0 {install} {configurations path}."
        exit -1
esac