#!/bin/bash

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

version=0.1
base_dir=../../
output_dir=./output

cwd=`pwd`

cd $base_dir
base_dir=`pwd`

cd $cwd

cd $output_dir
output_dir=`pwd`

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

        output_file=$output_dir/fate-$sub_module-$version.tar.gz
        echo "[INFO] $sub_module output_file: $output_file"

        rm -f $output_file
        tar czf $output_file lib fate-$sub_module-$version.jar
    done
    echo "--------------"
done


cd $cwd
