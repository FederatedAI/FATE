#!/bin/bash

version=0.3
cwd=`pwd`

cd ../../
fate_dir=`pwd`
base_dir=$fate_dir/arch
output_dir=$fate_dir/cluster-deploy/example-dir-tree

cd $cwd
source ./configurations.sh

cd $base_dir
targets=`find "$base_dir" -type d -name "target" -mindepth 2`

module="test"
sub_module="test"
for target in ${targets[@]}; do
    echo
    echo $target | awk -F "/" '{print $(NF - 2), $(NF - 1)}' | while read a b; do 
        module=$a
        sub_module=$b 

        cd $target

        jar_file="fate-$sub_module-$version.jar"
        if [[ ! -f $jar_file ]]; then
            echo "[INFO] $jar_file does not exist. skipping."
            continue
        fi

        output_file=$output_dir/$sub_module/fate-$sub_module-$version.tar.gz
        echo "[INFO] $sub_module output_file: $output_file"
		
		if [[ ! -d $output_dir/$sub_module ]]
		then
			break
		fi

        rm -f $output_file
        gtar czf $output_file lib fate-$sub_module-$version.jar
		cd $output_dir/$sub_module
		sed -i "s#JAVA_HOME=.*#JAVA_HOME=$javadir#g" ./service.sh
		tar -xzf fate-$sub_module-$version.tar.gz
		rm -f fate-$sub_module-$version.tar.gz
		ln -s fate-$sub_module-$version.jar fate-$sub_module.jar
    done
    echo "--------------"
done

cp -r $fate_dir/fate-serving/serving-server/target/lib $output_dir/serving-server/
cp $fate_dir/fate-serving/serving-server/target/fate-serving-server-$version.jar $output_dir/serving-server/
cd $output_dir/serving-server
ln -s fate-serving-server-$version.jar fate-serving-server.jar
sed -i "s#JAVA_HOME=.*#JAVA_HOME=$javadir#g" ./service.sh

cd $fate_dir
cp -r arch federatedml workflow examples $output_dir/python/
cp -r arch/eggroll/storage-service-cxx/* $output_dir/storage-service-cxx/
cd $output_dir/storage-service-cxx
mkdir logs
cd $cwd