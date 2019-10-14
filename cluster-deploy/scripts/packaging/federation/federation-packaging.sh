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
target_dir=$fate_dir/arch/driver/federation/target
#source build
source_build(){
#arch mvn
ping -c 4 www.baidu.com >>/dev/null 2>&1
if [ $? -eq 0 ];then
   echo "start execute mvn build"
   cd $fate_dir/arch &&  mvn clean package -DskipTests
           echo "arch mvn  build done"
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
        sed -i "s#JAVA_HOME=.*#JAVA_HOME=$javadir#g" ./service.sh
        tar -xzf fate-$sub_module-$version.tar.gz
        rm -f fate-$sub_module-$version.tar.gz
        if [[ ! -f "fate-$sub_module.jar" ]]; then
           ln -s fate-$sub_module-$version.jar fate-$sub_module.jar
        fi

  cd  $fate_dir/cluster-deploy/example-dir-tree
  tar -czf federation.tar  federation/*
  mv  federation.tar $output_dir
}
#build
build() {
return 0
}
#configurations
config() {
cd  $fate_dir/cluster-deploy/example-dir-tree
sed -i "18s/service.port=.*/service.port=9394/g" ./federation/conf/federation.properties
sed -i "s/meta.service.port=.*/meta.service.port=8590/g" ./federation/conf/federation.properties
sed -i "s/party.id=.*/party.id=$partyid/g" ./federation/conf/federation.properties
sed -i "s/meta.service.ip=.*/meta.service.ip=$mip/g" ./federation/conf/federation.properties
cd federation
tar  -czf federation-conf.tar  conf/*
mv federation-conf.tar $output_dir
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