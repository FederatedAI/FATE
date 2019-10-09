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
cdir=$redisdir
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
if [ ! -f "$basedir/packages/redis-$redisversion.tar.gz" ];then
    echo "No redis-$redisversion.tar.gz package can't be installed"
    exit
fi
tar -xzvf $basedir/packages/redis-$redisversion.tar.gz
cd redis*
mkdir bin
mkdir conf
make
cd src
sudo make install
cp redis-cli redis-server ../bin
}
#build
build(){
  return 0
}
#configuration
config(){
cd ..
cp redis.conf ./conf/
sed -i "s/127.0.0.1/0.0.0.0/g" ./conf/redis.conf
sed -i "s/# requirepass foobared/requirepass $redispass /g" ./conf/redis.conf
sed -i "s/databases 16/databases 50/g" ./conf/redis.conf
}
#init service
init(){
cd $cdir/redis*
cp $basedir/redis/service.sh $cdir/redis*
nohup bin/redis-server conf/redis.conf &
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
