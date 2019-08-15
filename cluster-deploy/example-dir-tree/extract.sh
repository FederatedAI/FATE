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
modules=(federation meta-service proxy roll storage-service )

cwd=`pwd`

for module in "${modules[@]}"; do
    tar_file="fate-${module}-${version}.tar.gz"
    if [[ -f ${tar_file} ]]; then
        echo "[INFO] extracting ${tar_file}"
        if [[ -n ${module} ]]; then
            cd ${module}
            rm *.jar
            rm -r lib
            tar xzf ${cwd}/${tar_file} -C .
            ln -sf fate-${module}-${version}.jar fate-${module}.jar
            cd ${cwd}
            rm ${tar_file}
        fi
    else
        echo "[INFO] no tar file for module ${module}"
    fi
done
