#!/bin/bash

set -e

if [[ $2 == "" ]]
then
echo "usage: $0 {package} {configurations path}."
exit
else
  source $2
fi
cwd=`pwd`
cd ../../
fate_dir=`pwd`
echo $fate_dir
#source build
source_build(){
ping -c 4 www.baidu.com >>/dev/null 2>&1
if [ $? -eq 0 ];then
echo "start execute mvn build"
cd $fate_dir/arch &&  mvn clean package -DskipTests
cd $fate_dir/fate-serving &&  mvn clean package -DskipTests
echo "mvn  build done"
else
echo "Sorry,the host cannot access the public network."
fi
cp -r $fate_dir/fate-serving/serving-server/target/lib $fate_dir/cluster-deploy/example-dir-tree/serving-server/
cp $fate_dir/fate-serving/serving-server/target/fate-serving-server-$fateversion.jar $fate_dir/cluster-deploy/example-dir-tree/serving-server/
cd $fate_dir/cluster-deploy/example-dir-tree/serving-server
if [[ ! -f "fate-serving-server.jar" ]]; then
ln -s fate-serving-server-$fateversion.jar fate-serving-server.jar
fi
cd ..
tar -czf serving-server.tar  serving-server/*
mv  serving-server.tar $output_dir
}
#build
build() {
   return 0
}
#configurations
config() {
cd $fate_dir/cluster-deploy/example-dir-tree
sed -i "s#JAVA_HOME=.*#JAVA_HOME=$javadir#g" ./serving-server/service.sh
sed -i "s/ip=.*/ip=$sip/g" ./serving-server/conf/serving-server.properties
sed -i "s/workMode=.*/workMode=1/g" ./serving-server/conf/serving-server.properties
sed -i "s/party.id=.*/party.id=$partyid/g" ./serving-server/conf/serving-server.properties
sed -i "s/port=8000/port=8001/g" ./serving-server/conf/serving-server.properties
sed -i "s/proxy=.*/proxy=$pip:9370/g" ./serving-server/conf/serving-server.properties
sed -i "s/roll=.*/roll=$rip:8011/g" ./serving-server/conf/serving-server.properties
sed -i "s/redis.ip=.*/redis.ip=$redisip/g" ./serving-server/conf/serving-server.properties
sed -i "s/redis.port=.*/redis.port=6379/g" ./serving-server/conf/serving-server.properties
sed -i "s/redis.password=.*/redis.password=$redispass/g" ./serving-server/conf/serving-server.properties
cd serving-server
tar  -czf serving-server-conf.tar  conf/*
mv serving-server-conf.tar $output_dir
}
init (){
        return 0
}

case "$1" in
    package)
            source_build
            build
            config
            init
        ;;
        *)
                echo "usage: $0 {package} {configurations path}."
        exit -1
esac