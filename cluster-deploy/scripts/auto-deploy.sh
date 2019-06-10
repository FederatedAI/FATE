#!/bin/bash

cwd=`pwd`

cd ../../
fate_dir=`pwd`
output_dir=$fate_dir/cluster-deploy/example-dir-tree

cd $cwd
source ./configurations.sh
cp ./modify_json.py $output_dir/python/

cd $output_dir
sed -i "s#PATH=.*#PATH=$dir/python#g" ./python/processor.sh
sed -i "s#src/arch/processor#arch/processor#g" ./python/processor.sh
sed -i "s#JAVA_HOME=.*#JAVA_HOME=$javadir#g" ./python/service.sh
sed -i "s#venv=.*#venv=$venvdir#g" ./python/service.sh
sed -i "18s/service.port=.*/service.port=9394/g" ./federation/conf/federation.properties
sed -i "s/meta.service.port=.*/meta.service.port=8590/g" ./federation/conf/federation.properties
sed -i "s#/jdbc.driver.classname.*#jdbc.driver.classname=com.mysql.cj.jdbc.Driver#g" ./meta-service/conf/jdbc.properties
sed -i "s/target.project=.*/target.project=meta-service/g" ./meta-service/conf/jdbc.properties
sed -i "s/port=.*/port=9370/g" ./proxy/conf/proxy.properties
sed -i "s#route.table=.*#route.table=$dir/proxy/conf/route_table.json#g" ./proxy/conf/proxy.properties
sed -i "s/service.port=.*/service.port=8011/g" ./roll/conf/roll.properties
sed -i "s/meta.service.port=.*/meta.service.port=8590/g" ./roll/conf/roll.properties
sed -i "s/service.port=.*/service.port=7888/g" ./egg/conf/egg.properties
sed -i "s#processor.venv=.*#processor.venv=$venvdir#g" ./egg/conf/egg.properties
sed -i "s#processor.path=.*#processor.path=$dir/python/arch/processor/processor.py#g" ./egg/conf/egg.properties
sed -i "s#python.path=.*#python.path=$dir/python#g" ./egg/conf/egg.properties
sed -i "s#data.dir=.*#data.dir=$dir/data-dir#g" ./egg/conf/egg.properties
sed -i "s/max.processors.count=.*/max.processors.count=16/g" ./egg/conf/egg.properties
sed -i "s/IP =.*/IP = \'0.0.0.0\'/g" ./python/arch/task_manager/settings.py
sed -i "s/WORK_MODE =.*/WORK_MODE = 1/g" ./python/arch/task_manager/settings.py
sed -i "s#PYTHONPATH=.*#PYTHONPATH=$dir/python#g" ./python/arch/task_manager/service.sh
sed -i "s#venv=.*#venv=$venvdir#g" ./python/arch/task_manager/service.sh
sed -i "20s#-I. -I.*#-I. -I$dir/storage-service-cxx/third_party/include#g" ./storage-service-cxx/Makefile
sed -i "34s#LDFLAGS += -L.*#LDFLAGS += -L$dir/storage-service-cxx/third_party/lib -llmdb -lboost_system -lboost_filesystem -lglog -lgpr#g" ./storage-service-cxx/Makefile
sed -i "36s#PROTOC =.*#PROTOC = $dir/storage-service-cxx/third_party/bin/protoc#g" ./storage-service-cxx/Makefile
sed -i "37s#GRPC_CPP_PLUGIN =.*#GRPC_CPP_PLUGIN = $dir/storage-service-cxx/third_party/bin/grpc_cpp_plugin#g" ./storage-service-cxx/Makefile
sed -i "s#/usr/local/lib.*#/usr/local/lib:$dir/storage-service-cxx/third_party/lib#g" ./storage-service-cxx/service.sh

tar -czf fate.tar ./*

eval iplength=\${#iplist[*]}
	for ((j=0;j<$iplength;j++))
	do
		eval ip=\${iplist[${j}]}
		echo "$eip copy is ok!"
		scp fate.tar $user@$ip:$dir
		ssh -tt $user@$ip<< eeooff
cd $dir
tar -xzf fate.tar
rm -f fate.tar
exit
eeooff
	done

if [ $exchangeip ]
then
	scp fate.tar $user@$exchangeip:$dir
fi
for ((i=0;i<${#partylist[*]};i++))
do
	eval fip=\${fedlist${i}[0]}
	eval mip=\${meta${i}[0]}
	eval pip=\${proxy${i}[0]}
	eval rip=\${roll${i}[0]}
	eval sip1=\${serving${i}[0]}
	eval sip2=\${serving${i}[1]}
	eval tmip=\${tmlist${i}[0]}
	eval partyid=\${partylist[${i}]}
	eval jdbcip=\${JDBC${i}[0]}
	eval jdbcdbname=\${JDBC${i}[1]}
	eval jdbcuser=\${JDBC${i}[2]}
	eval jdbcpasswd=\${JDBC${i}[3]}
	eval elength=\${#egglist${i}[*]}
	eval slength=\${#serving${i}[*]}
	
if [ ! $exchangeip ]
then
	if [ $(($i%2)) == 0 ]
	then
		eval exchangeip=\${proxy0[0]}
	else
		eval exchangeip=\${proxy1[0]}
	fi
else
	echo exchangeip=$exchangeip
	if ssh -tt $user@$exchangeip test -e $dir/proxy;then
		ssh -tt $user@$exchangeip << eeooff
cd $dir
rm -f fate.tar
export PYTHONPATH=/data/projects/fate/python
source $venvdir/bin/activate
sed -i "s/ip=.*/ip=/g" ./proxy/conf/proxy.properties
sed -i "s/partyId=.*/partyId=\"$partyid\"/g" ./python/modify_json.py
sed -i "s/pip=.*/pip=\"$pip\"/g" ./python/modify_json.py
python python/modify_json.py exchange ./proxy/conf/route_table.json
exit
eeooff
	else
		ssh -tt $user@$exchangeip << eeooff
cd $dir
tar -xzf fate.tar
rm -f fate.tar
export PYTHONPATH=/data/projects/fate/python
source $venvdir/bin/activate
sed -i '3,10d' ./proxy/conf/route_table.json
sed -i "s/ip=.*/ip=/g" ./proxy/conf/proxy.properties
sed -i "s/partyId=.*/partyId=\"$partyid\"/g" ./python/modify_json.py
sed -i "s/pip=.*/pip=\"$pip\"/g" ./python/modify_json.py
python python/modify_json.py exchange ./proxy/conf/route_table.json
exit
eeooff
	fi
fi

	ssh -tt $user@$fip << eeooff
cd $dir
sed -i "s/party.id=.*/party.id=$partyid/g" ./federation/conf/federation.properties
sed -i "s/meta.service.ip=.*/meta.service.ip=$mip/g" ./federation/conf/federation.properties
exit
eeooff
	echo federation module of $partyid done!
	ssh -tt $user@$mip << eeooff
cd $dir
sed -i "s/party.id=.*/party.id=$partyid/g" ./meta-service/conf/meta-service.properties
sed -i "s#//.*?#//localhost:3306/$jdbcdbname?#g" ./meta-service/conf/jdbc.properties
sed -i "s/jdbc.username=.*/jdbc.username=$jdbcuser/g" ./meta-service/conf/jdbc.properties
sed -i "s/jdbc.password=.*/jdbc.password=$jdbcpasswd/g" ./meta-service/conf/jdbc.properties
exit
eeooff
	echo meta-service module of $partyid done!
	ssh -tt $user@$pip << eeooff
cd $dir
export PYTHONPATH=/data/projects/fate/python
source $venvdir/bin/activate
sed -i "s/coordinator=.*/coordinator=$partyid/g" ./proxy/conf/proxy.properties
sed -i "s/ip=.*/ip=$pip/g" ./proxy/conf/proxy.properties
sed -i "s/exchangeip=.*/exchangeip=\"$exchangeip\"/g" ./python/modify_json.py
sed -i "s/fip=.*/fip=\"$fip\"/g" ./python/modify_json.py
sed -i "s/tmip=.*/tmip=\"$tmip\"/g" ./python/modify_json.py
sed -i "s/sip1=.*/sip1=\"$sip1\"/g" ./python/modify_json.py
sed -i "s/sip2=.*/sip2=\"$sip2\"/g" ./python/modify_json.py
sed -i "s/partyId=.*/partyId=\"$partyid\"/g" ./python/modify_json.py
python python/modify_json.py proxy ./proxy/conf/route_table.json
exit
eeooff
	echo proxy module of $partyid done!
	ssh -tt $user@$rip << eeooff
cd $dir
sed -i "s/party.id=.*/party.id=$partyid/g" ./roll/conf/roll.properties
sed -i "s/meta.service.ip=.*/meta.service.ip=$mip/g" ./roll/conf/roll.properties
exit
eeooff
	echo roll module of $partyid done!
	for ((c=0;c<$slength;c++))
	do
		eval sip=\${serving${i}[${c}]}
		ssh -tt $user@$sip << eeooff
cd $dir
sed -i "s/ip=.*/ip=$sip/g" ./serving-server/conf/serving-server.properties
sed -i "s/workMode=.*/workMode=1/g" ./serving-server/conf/serving-server.properties
sed -i "s/party.id=.*/party.id=$partyid/g" ./serving-server/conf/serving-server.properties
sed -i "s/port=8000/port=8001/g" ./serving-server/conf/serving-server.properties
sed -i "s/proxy=.*/proxy=$pip:9370/g" ./serving-server/conf/serving-server.properties
sed -i "s/roll=.*/roll=$rip:8011/g" ./serving-server/conf/serving-server.properties
sed -i "s/redis.ip=.*/redis.ip=0.0.0.0/g" ./serving-server/conf/serving-server.properties
sed -i "s/redis.port=.*/redis.port=6379/g" ./serving-server/conf/serving-server.properties
sed -i "s/redis.password=.*/redis.password=$redispass/g" ./serving-server/conf/serving-server.properties
exit
eeooff
	done
	echo serving module of $partyid done!
	for ((a=0;a<$elength;a++))
	do
		eval eip=\${egglist${i}[${a}]}
		ssh -tt $user@$eip << eeooff
cd $dir
export PYTHONPATH=/data/projects/fate/python 
source $venvdir/bin/activate
sed -i "s/party.id=.*/party.id=$partyid/g" ./egg/conf/egg.properties
sed -i "s/fip=.*/fip=\"$fip\"/g" ./python/modify_json.py
sed -i "s/rip=.*/rip=\"$rip\"/g" ./python/modify_json.py
sed -i "s/pip=.*/pip=\"$pip\"/g" ./python/modify_json.py
sed -i "s/sip1=.*/sip1=\"$sip1\"/g" ./python/modify_json.py
sed -i "s/sip2=.*/sip2=\"$sip2\"/g" ./python/modify_json.py
sed -i "s/tmip=.*/tmip=\"$tmip\"/g" ./python/modify_json.py
sed -i "s/partyId=.*/partyId=\"$partyid\"/g" ./python/modify_json.py
python python/modify_json.py python ./python/arch/conf/server_conf.json	
sed -i "s/PARTY_ID =.*/PARTY_ID = \"$partyid\"/g" ./python/arch/task_manager/settings.py
sed -i "s/'user':.*/'user': '$jdbcuser',/g" ./python/arch/task_manager/settings.py
sed -i "s/'passwd':.*/'passwd': '$jdbcpasswd',/g" ./python/arch/task_manager/settings.py
sed -i "s/'host':.*/'host': '$jdbcip',/g" ./python/arch/task_manager/settings.py
sed -i "s/localhost/$tmip/g" ./python/arch/task_manager/settings.py
sudo su root
cd $dir/storage-service-cxx
cd third_party/boost
sed -i "14s#PREFIX=.*#PREFIX=$dir/storage-service-cxx/third_party#g" ./bootstrap.sh 
./bootstrap.sh
./b2 install

cd ../glog
./autogen.sh
./configure  --prefix=$dir/storage-service-cxx/third_party
make && make install

cd ../grpc
make
mkdir -p $dir/third_party
make install prefix=$dir/third_party
cd third_party/protobuf
./autogen.sh
./configure --prefix=$dir/third_party
make
make check
make install
cd $dir/storage-service-cxx/
rsync -a $dir/third_party/* ./third_party/

cd third_party/lmdb/libraries/liblmdb
make
cp lmdb.h $dir/storage-service-cxx/third_party/include
cp liblmdb.so $dir/storage-service-cxx/third_party/lib

cd $dir 
rm -rf third_party
chown -R app:apps ./*

exit
cd $dir/storage-service-cxx/
make

exit
eeooff
	done
	echo egg and task_manager module of $partyid done!
	ssh -tt $user@$jdbcip<< eeooff
sed -i "s/eggroll_meta/$jdbcdbname/g" $dir/python/arch/eggroll/meta-service/src/main/resources/create-meta-service.sql
${mysqldir}/bin/mysql -u$jdbcuser -p$jdbcpasswd -S ${mysqldir}/mysql.sock
create database task_manager;
source $dir/python/arch/eggroll/meta-service/src/main/resources/create-meta-service.sql
INSERT INTO node (ip, port, type, status) values ('${rip}', '8011', 'ROLL', 'HEALTHY');
INSERT INTO node (ip, port, type, status) values ('${pip}', '9370', 'PROXY', 'HEALTHY');
exit;
exit
eeooff
	for ((b=0;b<$elength;b++))
	do
		eval eip=\${egglist${i}[${b}]}
		ssh -tt $user@$jdbcip<< eeooff
${mysqldir}/bin/mysql -u$jdbcuser -p$jdbcpasswd -S ${mysqldir}/mysql.sock
use $jdbcdbname;
INSERT INTO node (ip, port, type, status) values ('${eip}', '7888', 'EGG', 'HEALTHY');
INSERT INTO node (ip, port, type, status) values ('${eip}', '7778', 'STORAGE', 'HEALTHY');
exit;
exit
eeooff
	done
done
cd $cwd