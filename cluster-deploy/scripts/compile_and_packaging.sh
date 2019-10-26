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

set -e
module_name="egg"
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source_code_dir=$(cd `dirname ${cwd}`; cd ../../; pwd)
echo ${source_code_dir}
packages_dir=${source_code_dir}/cluster-deploy/packages
mkdir -p ${packages_dir}

echo "[INFO] Packaging eggroll"
eggroll_source_code_dir=${source_code_dir}/eggroll
cd ${eggroll_source_code_dir}
echo "[INFO] Compiling eggroll"
mvn clean package -DskipTests
echo "[INFO] Compile eggroll done"

cd ${eggroll_source_code_dir}
cd api
tar czf eggroll-api-1.1.tar.gz *
mv eggroll-api-1.1.tar.gz ${packages_dir}/

cd ${eggroll_source_code_dir}
cd computing
tar czf eggroll-computing-1.1.tar.gz *
mv eggroll-computing-1.1.tar.gz ${packages_dir}/

cd ${eggroll_source_code_dir}
cd conf
tar czf eggroll-conf-1.1.tar.gz *
mv eggroll-conf-1.1.tar.gz ${packages_dir}/

cd ${eggroll_source_code_dir}
cd framework/egg/target
tar czf eggroll-egg-1.1.tar.gz eggroll-egg-1.1.jar lib/
mv eggroll-egg-1.1.jar ${packages_dir}/

cd ${eggroll_source_code_dir}
cd framework/meta-service/target
tar czf eggroll-meta-service-1.1.tar.gz eggroll-meta-service-1.1.jar lib/
mv eggroll-meta-service-1.1.tar.gz ${packages_dir}/

cd ${eggroll_source_code_dir}
cd framework/roll/target
tar czf eggroll-roll-1.1.tar.gz eggroll-roll-1.1.jar lib/
mv eggroll-roll-1.1.tar.gz ${packages_dir}/

cd ${eggroll_source_code_dir}
cd storage/storage-service-cxx
tar czf eggroll-storage-service-cxx-1.1.tar.gz *
mv eggroll-storage-service-cxx-1.1.tar.gz ${packages_dir}/
echo "[INFO] Package eggroll done"

echo "[INFO] Packaging fate"
echo "[INFO] Compiling fate"
cd ${source_code_dir}/fateboard/
mvn clean package -DskipTests
cd ${source_code_dir}/arch/
mvn clean package -DskipTests
echo "[INFO] Compile fate done"
cp ${source_code_dir}/fateboard/target/fateboard-1.1.jar ${packages_dir}/

cd ${source_code_dir}/arch/driver/federation/target
tar czf fate-federation-1.1.tar.gz fate-federation-1.1.jar lib/
mv fate-federation-1.1.tar.gz ${packages_dir}/

cd ${source_code_dir}/arch/networking/proxy/target
tar czf fate-proxy-1.1.tar.gz fate-proxy-1.1.jar lib/
mv fate-proxy-1.1.tar.gz ${packages_dir}/
