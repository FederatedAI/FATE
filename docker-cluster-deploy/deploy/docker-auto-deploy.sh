########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

#!/bin/bash

BASEDIR=$(dirname "$0")
cd $BASEDIR
WORKINGDIR=`pwd`

# fetch fate-python image
source ${WORKINGDIR}/../.env
source ${WORKINGDIR}/docker-configuration.sh

cp ${WORKINGDIR}/../../cluster-deploy/scripts/modify_json.py ${WORKINGDIR}/docker-example-dir-tree/python/

cd ${WORKINGDIR}/docker-example-dir-tree
sed -ie "s#PATH=.*#PATH=$dir/python#g" ./python/processor.sh
sed -ie "s#src/arch/processor#arch/processor#g" ./python/processor.sh
sed -ie "18s/service.port=.*/service.port=9394/g" ./federation/conf/federation.properties
sed -ie "s/meta.service.port=.*/meta.service.port=8590/g" ./federation/conf/federation.properties
sed -ie "s#/jdbc.driver.classname.*#jdbc.driver.classname=com.mysql.cj.jdbc.Driver#g" ./meta-service/conf/jdbc.properties
sed -ie "s/target.project=.*/target.project=meta-service/g" ./meta-service/conf/jdbc.properties
sed -ie "s/port=.*/port=9370/g" ./proxy/conf/proxy.properties
sed -ie "s#route.table=.*#route.table=$dir/proxy/conf/route_table.json#g" ./proxy/conf/proxy.properties
sed -ie "s/service.port=.*/service.port=8011/g" ./roll/conf/roll.properties
sed -ie "s/meta.service.port=.*/meta.service.port=8590/g" ./roll/conf/roll.properties
sed -ie "s/service.port=.*/service.port=7888/g" ./egg/conf/egg.properties
sed -ie "s#processor.venv=.*#processor.venv=$venvdir#g" ./egg/conf/egg.properties
sed -ie "s#processor.path=.*#processor.path=$dir/python/arch/processor/processor.py#g" ./egg/conf/egg.properties
sed -ie "s#python.path=.*#python.path=$dir/python#g" ./egg/conf/egg.properties
sed -ie "s#data.dir=.*#data.dir=$dir/data-dir#g" ./egg/conf/egg.properties
sed -ie "s/max.processors.count=.*/max.processors.count=16/g" ./egg/conf/egg.properties


cd ${WORKINGDIR}

CONTAINER_ID=`docker run -d ${PREFIX}/python:${TAG}`
for ((i=0;i<${#partylist[*]};i++))
do
  eval fip=federation
  eval flip=python
  eval fbip=fateboard
  eval mip=meta-service
  eval pip=proxy
  eval rip=roll
  eval redisip=redis
  eval redispass=fate1234
  eval sip1=serving-server
  eval tmip=python
  eval partyid=\${partylist[${i}]}
  eval partyip=\${partyiplist[${i}]}
  eval jdbcip=mysql
  eval jdbcdbname=fate
  eval jdbcuser=fate
  eval jdbcpasswd=Fate123#$
  eval elength=egg
  eval slength=serving-server
    
  mkdir -p confs-$partyid/confs
  cp -r docker-example-dir-tree/* confs-$partyid/confs/
  cp ./docker-compose.yml confs-$partyid/
  
  sed -ie "s/party.id=.*/party.id=$partyid/g" ./confs-$partyid/confs/federation/conf/federation.properties
  sed -ie "s/meta.service.ip=.*/meta.service.ip=$mip/g" ./confs-$partyid/confs/federation/conf/federation.properties
  echo federation module of $partyid done!
  
  sed -ie "s/party.id=.*/party.id=$partyid/g" ./confs-$partyid/confs/meta-service/conf/meta-service.properties
  sed -ie "s#//.*?#//mysql:3306/$jdbcdbname?#g" ./confs-$partyid/confs/meta-service/conf/jdbc.properties
  sed -ie "s/jdbc.username=.*/jdbc.username=$jdbcuser/g" ./confs-$partyid/confs/meta-service/conf/jdbc.properties
  sed -ie "s/jdbc.password=.*/jdbc.password=$jdbcpasswd/g" ./confs-$partyid/confs/meta-service/conf/jdbc.properties
  echo meta-service module of $partyid done!

  sed -ie "s#^fate.url=.*#fate.url=http://python:9380#g" ./confs-$partyid/confs/fateboard/conf/application.properties
  sed -i "s#^spring.datasource.url=.*#spring.datasource.url=jdbc:mysql://mysql:3306/fate?characterEncoding=utf8\&characterSetResults=utf8\&autoReconnect=true\&failOverReadOnly=false\&serverTimezone=GMT%2B8#g" ./confs-$partyid/confs/fateboard/conf/application.properties
  sed -i "s/^spring.datasource.username=.*/spring.datasource.username=$jdbcuser/g" ./confs-$partyid/confs/fateboard/conf/application.properties
  sed -i "s/^spring.datasource.password=.*/spring.datasource.password=$jdbcpasswd/g" ./confs-$partyid/confs/fateboard/conf/application.properties
  echo fateboard module of $partyid done!

  sed -ie "s/coordinator=.*/coordinator=$partyid/g" ./confs-$partyid/confs/proxy/conf/proxy.properties
  sed -ie "s/ip=.*/ip=0.0.0.0/g" ./confs-$partyid/confs/proxy/conf/proxy.properties
  sed -ie "s/exchangeip=.*/exchangeip=\"$exchangeip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/fip=.*/fip=\"$fip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/tmip=.*/tmip=\"$tmip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/flip=.*/flip=\"$flip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/fbip=.*/fbip=\"$fbip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "14s/sip1=.*/sip1=\"$sip1\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/partyId=.*/partyId=\"$partyid\"/g" ./confs-$partyid/confs/python/modify_json.py

  echo proxy module of $partyid done!
  
  sed -ie "s/party.id=.*/party.id=$partyid/g" ./confs-$partyid/confs/roll/conf/roll.properties
  sed -ie "s/meta.service.ip=.*/meta.service.ip=$mip/g" ./confs-$partyid/confs/roll/conf/roll.properties
  echo roll module of $partyid done!
  
  sed -ie "s/ip=.*/ip=$sip1/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/workMode=.*/workMode=1/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/party.id=.*/party.id=$partyid/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/port=8000/port=8001/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/proxy=.*/proxy=$pip:9370/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/roll=.*/roll=$rip:8011/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/redis.ip=.*/redis.ip=$redisip/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/redis.port=.*/redis.port=6379/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  sed -ie "s/redis.password=.*/redis.password=$redispass/g" ./confs-$partyid/confs/serving-server/conf/serving-server.properties
  echo serving module of $partyid done!
  
  sed -ie "s/IP =.*/IP = \'0.0.0.0\'/g" ./confs-$partyid/confs/python/fate_flow/settings.py
  sed -ie "s/WORK_MODE =.*/WORK_MODE = 1/g" ./confs-$partyid/confs/python/fate_flow/settings.py
  # sed -ie "s#PYTHONPATH=.*#PYTHONPATH=$dir/python#g" ./confs-$partyid/confs/fate_flow/service.sh
  # sed -ie "s#venv=.*#venv=$venvdir#g" ./confs-$partyid/confs/fate_flow/service.sh
  sed -ie "s/PARTY_ID =.*/PARTY_ID = \"$partyid\"/g" ./confs-$partyid/confs/python/fate_flow/settings.py
  sed -ie "s/'user':.*/'user': '$jdbcuser',/g" ./confs-$partyid/confs/python/fate_flow/settings.py
  sed -ie "s/'passwd':.*/'passwd': '$jdbcpasswd',/g" ./confs-$partyid/confs/python/fate_flow/settings.py
  sed -ie "56s/'host':.*/'host': '$jdbcip',/g" ./confs-$partyid/confs/python/fate_flow/settings.py
  sed -ie "65s/'password':.*/'password': '$redispass',/g" ./confs-$partyid/confs/python/fate_flow/settings.py
  sed -ie "s/localhost/0.0.0.0/g" ./confs-$partyid/confs/python/fate_flow/settings.py

  # generate conf dir
  mkdir -p ./confs-$partyid/confs/python/arch/conf
  cp ${WORKINGDIR}/../.env ./confs-$partyid
  cp ./confs-$partyid/confs/python/fate_flow/settings.py ./confs-$partyid/confs/python/arch/conf/settings.py

  sed -ie "s/party.id=.*/party.id=$partyid/g" ./confs-$partyid/confs/egg/conf/egg.properties
  sed -ie "s/fip=.*/fip=\"$fip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/rip=.*/rip=\"$rip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/pip=.*/pip=\"$pip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "11s/sip1=.*/sip1=\"$sip1\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/tmip=.*/tmip=\"$tmip\"/g" ./confs-$partyid/confs/python/modify_json.py
  sed -ie "s/partyId=.*/partyId=\"$partyid\"/g" ./confs-$partyid/confs/python/modify_json.py
  echo egg and fate_flow module of $partyid done!
  
  # modify_json.py as input
  docker cp ./confs-$partyid/confs/python/modify_json.py ${CONTAINER_ID}:/data/projects/fate/modify_json.py
  docker cp ./confs-$partyid/confs/python/conf/server_conf.json ${CONTAINER_ID}:/data/projects/fate/server_conf.json
  docker cp ./confs-$partyid/confs/proxy/conf/route_table.json ${CONTAINER_ID}:/data/projects/fate/route_table.json

  echo "Generating configuration files within container"
  docker exec ${CONTAINER_ID} /bin/sh -c "python /data/projects/fate/modify_json.py proxy /data/projects/fate/route_table.json;"
  docker exec ${CONTAINER_ID} /bin/sh -c "python /data/projects/fate/modify_json.py python /data/projects/fate/server_conf.json;"

  # Add info of other parties to route_table
  for ((j=0;j<${#partylist[*]};j++))
  do
    if [ "${partylist[${j}]}" != "${partyid}" ]
    then
      sed -ie "s/partyId=.*/partyId=\"${partylist[${j}]}\"/g" ./confs-$partyid/confs/python/modify_json.py
      sed -ie "s/pip=.*/pip=\"${partyiplist[${j}]}\"/g" ./confs-$partyid/confs/python/modify_json.py
      docker cp ./confs-$partyid/confs/python/modify_json.py ${CONTAINER_ID}:/data/projects/fate/modify_json.py
      docker exec ${CONTAINER_ID} /bin/sh -c "python /data/projects/fate/modify_json.py exchange /data/projects/fate/route_table.json;"
    fi
  done
  # route_table.json and server_conf.json as outputs
  docker cp ${CONTAINER_ID}:/data/projects/fate/route_table.json ./confs-$partyid/confs/proxy/conf/route_table.json
  docker cp ${CONTAINER_ID}:/data/projects/fate/server_conf.json ./confs-$partyid/confs/python/arch/conf/server_conf.json

  tar -czf ./confs-$partyid.tar ./confs-$partyid
  scp confs-$partyid.tar $user@$partyip:~/
  echo "$ip copy is ok!"
  ssh -tt $user@$partyip<< eeooff
mkdir -p $dir
mv ~/confs-$partyid.tar $dir
cd $dir
tar -xzf confs-$partyid.tar
rm -f confs-$partyid.tar 
cd confs-$partyid
docker-compose -f docker-compose.yml up -d
exit
eeooff
  rm -f confs-$partyid.tar 
  echo "party $partyid deploy is ok!"

done
docker stop ${CONTAINER_ID}
docker rm ${CONTAINER_ID}
