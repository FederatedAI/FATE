#!/bin/bash
base="$(cd `dirname $0`; pwd)"
cdir="/data/projects/common/redis"
mkdir -p $cdir
cd $cdir
tar -xzvf $base/packages/redis-5.0.2.tar.gz
cd redis*
mkdir bin
mkdir conf
make
cd src
sudo make install
cp redis-cli redis-server ../bin
cd ..
cp redis.conf ./conf/
sed -i 's/127.0.0.1/0.0.0.0/g' ./conf/redis.conf
sed -i 's/# requirepass foobared/requirepass fate1234/g' ./conf/redis.conf
sed -i 's/databases 16/databases 50/g' ./conf/redis.conf
nohup ./bin/redis-server ./conf/redis.conf &

