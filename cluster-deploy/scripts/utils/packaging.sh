#!/bin/bash

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
cd ${source_code_dir}/fateboard/
