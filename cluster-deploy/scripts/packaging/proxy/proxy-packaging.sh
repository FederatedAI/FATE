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
target_dir=$fate_dir/arch/networking/proxy/target
#source build
source_build(){
        ping -c 4 www.baidu.com >>/dev/null 2>&1
        if [ $? -eq 0 ];then
            echo "start execute mvn build"
            cd $fate_dir/arch &&  mvn clean package -DskipTests
            echo "arch mvn build done"
        else
            echo "Sorry,the host cannot access the public network."
       fi
      #packing
       module=`echo $target_dir | awk -F "/" '{print $(NF - 2), $(NF - 1)}' | awk '{print $1}'`
       sub_module=`echo $target_dir | awk -F "/" '{print $(NF - 2), $(NF - 1)}' | awk '{print $2}'`
       echo $module
       echo $sub_module
       cd $target_dir
       jar_file="fate-$sub_module-$version.jar"
        if [[ ! -f $jar_file ]]; then
            echo "[INFO] $jar_file does not exist. skipping."
            continue
        fi
        output_file=$fate_dir/cluster-deploy/example-dir-tree/$sub_module/fate-$sub_module-$version.tar.gz
        echo "[INFO] $sub_module output_file: $output_file"
        if [[ ! -d $fate_dir/cluster-deploy/example-dir-tre/$sub_module ]]
        then
                break
        fi
        rm -f $output_file
        gtar czf $output_file lib fate-$sub_module-$version.jar
        cd $fate_dir/cluster-deploy/example-dir-tree/$sub_module
        tar -xzf fate-$sub_module-$version.tar.gz
        rm -f fate-$sub_module-$version.tar.gz
        if [[ ! -f "fate-$sub_module.jar" ]]; then
           ln -s fate-$sub_module-$version.jar fate-$sub_module.jar
        fi
cd $fate_dir
cp -r arch federatedml workflow examples $fate_dir/cluster-deploy/example-dir-tree/python/
cp -r fate_flow  $fate_dir/cluster-deploy/example-dir-tree/python/
  cd  $cwd && cp ./modify_json.py $fate_dir/cluster-deploy/example-dir-tree/python/
  cd $fate_dir/cluster-deploy/example-dir-tree
 
  tar -czf proxy.tar proxy/*
  tar -czf python.tar python/*
  mv proxy.tar $output_dir
  mv python.tar $output_dir 
}
#build
build() {
        return 0
}
#configurations
config() {
  cd $fate_dir/cluster-deploy/example-dir-tree
  sed -i "s#PATH=.*#PATH=$dir/python#g" ./python/processor.sh
  sed -i "s#src/arch/processor#arch/processor#g" ./python/processor.sh
  sed -i "s#JAVA_HOME=.*#JAVA_HOME=$javadir#g" ./python/service.sh
  sed -i "s#venv=.*#venv=$venvdir#g" ./python/service.sh
  sed -i "s#JAVA_HOME=.*#JAVA_HOME=$javadir#g" ./proxy/service.sh

  sed -i "s/port=.*/port=9370/g" ./proxy/conf/proxy.properties
  sed -i "s#route.table=.*#route.table=$dir/proxy/conf/route_table.json#g" ./proxy/conf/proxy.properties
  
  sed -i "s/IP =.*/IP = \'0.0.0.0\'/g" ./python/fate_flow/settings.py
  sed -i "s/WORK_MODE =.*/WORK_MODE = 1/g" ./python/fate_flow/settings.py
  sed -i "s#PYTHONPATH=.*#PYTHONPATH=$dir/python#g" ./python/fate_flow/service.sh
  sed -i "s#venv=.*#venv=$venvdir#g" ./python/fate_flow/service.sh

export PYTHONPATH=/data/projects/fate/python
source $venvdir/bin/activate
sed -i "s/coordinator=.*/coordinator=$partyid/g" ./proxy/conf/proxy.properties
sed -i "s/ip=.*/ip=$pip/g" ./proxy/conf/proxy.properties
sed -i "s/exchangeip=.*/exchangeip=\"$exchangeip\"/g" ./python/modify_json.py
sed -i "s/fip=.*/fip=\"$fip\"/g" ./python/modify_json.py
sed -i "s/flip=.*/flip=\"$flip\"/g" ./python/modify_json.py
sed -i "s/sip1=.*/sip1=\"$sip1\"/g" ./python/modify_json.py
sed -i "s/sip2=.*/sip2=\"$sip2\"/g" ./python/modify_json.py
sed -i "s/partyId=.*/partyId=\"$partyid\"/g" ./python/modify_json.py
python python/modify_json.py proxy ./proxy/conf/route_table.json
cd proxy
tar  -czf proxy-conf.tar  conf/*
mv proxy-conf.tar $output_dir
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
