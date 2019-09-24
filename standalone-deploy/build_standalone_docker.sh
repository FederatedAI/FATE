#!/usr/bin/env bash

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

set -x

wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate.tar.gz

tar -xf fate.tar.gz

rm -rf init.sh requirments.txt requirements.txt

cp ../requirements.txt ./docker/python

if [ `uname -s` == "Darwin" ]; then
    sed -i ".bak" "s/'user':.*/'user': 'fate_dev',/g" ./fate_flow/settings.py
    sed -i ".bak" "s/'passwd':.*/'passwd': 'fate_dev',/g" ./fate_flow/settings.py
    sed -i ".bak" "s/'host':.*/'host': 'mysql',/g" ./fate_flow/settings.py
else
    sed -i "s/'user':.*/'user': 'fate_dev',/g" ./fate_flow/settings.py
    sed -i "s/'passwd':.*/'passwd': 'fate_dev',/g" ./fate_flow/settings.py
    sed -i "s/'host':.*/'host': 'mysql',/g" ./fate_flow/settings.py
fi

tar -cf ./docker/python/fate.tar arch federatedml workflow examples fate_flow research

logPath="/var/lib/fate/log"
if [ ! -d "$logPath" ]; then
   mkdir -p "$logPath"
fi

dataPath="/var/lib/fate/data"
if [ ! -d "$dataPath" ]; then
   mkdir -p "$dataPath"
fi
cp -r ./fate_flow/* /var/lib/fate/data

if [ `uname -s` == "Darwin" ]; then
    sed -i ".bak" "s#^fateflow.url=.*#fateflow.url=http://python:9380#g" ./fateboard/conf/application.properties
    sed -i ".bak" "s#^spring.datasource.url=.*#spring.datasource.url=jdbc:sqlite:/fate/fate_flow/fate_flow_sqlite.db#g" ./fateboard/conf/application.properties
    # sed -i ".bak" "s/^spring.datasource.username=.*/spring.datasource.username=fate_dev/g" ./fateboard/conf/application.properties
    # sed -i ".bak" "s/^spring.datasource.password=.*/spring.datasource.password=fate_dev/g" ./fateboard/conf/application.properties
else
    sed -i  "s#^fateflow.url=.*#fateflow.url=http://python:9380#g" ./fateboard/conf/application.properties
    sed -i "s#^spring.datasource.url=.*#spring.datasource.url=jdbc:sqlite:/fate/fate_flow/fate_flow_sqlite.db#g" ./fateboard/conf/application.properties
    # sed -i "s/^spring.datasource.username=.*/spring.datasource.username=fate_dev/g" ./fateboard/conf/application.properties
    # sed -i "s/^spring.datasource.password=.*/spring.datasource.password=fate_dev/g" ./fateboard/conf/application.properties
fi

cat ./fateboard/conf/application.properties

cd fateboard
ln -s fateboard-1.0.jar fateboard.jar
cd ..
tar -cf ./docker/fateboard/fateboard.tar fateboard

docker-compose -f ./docker/docker-compose-build.yml up -d

#sleep 15
docker restart fate_python
sleep 5
docker restart fate_fateboard

rm -rf examples workflow arch federatedml fateboard fate_flow research fate.tar.gz data

rm docker/python/fate.tar

rm docker/python/requirements.txt

rm docker/fateboard/fateboard.tar