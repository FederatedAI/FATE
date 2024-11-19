#!/bin/bash

# Copyright 2019-2022 VMware, Inc.
# Copyright 2019 The FATE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
BASEDIR=$(dirname "$0")
cd "$BASEDIR"
WORKINGDIR=$(pwd)
deploy_dir=/data/projects/fate

# fetch fate-python image
source "${WORKINGDIR}"/.env
source "${WORKINGDIR}"/parties.conf

echo "Generate Config"
echo "Info:"
echo "  KubeFATE Version: ${KubeFATE_Version}"
echo "  RegistryURI: ${RegistryURI}"
echo "  Computing: ${computing}"
echo "  Federation: ${federation}"
echo "  Storage: ${storage}"
echo "  Algorithm: ${algorithm}"
echo "  Device: ${device}"
echo "  Compute_core: ${compute_core}"

echo "  Party_List:"
for ((i = 0; i < ${#party_list[*]}; i++)); do
	echo "  - Party_ID:${party_list[${i}]}"
	echo "    Party_IP:${party_ip_list[${i}]}"
done
echo ""
echo ""
cd ${WORKINGDIR}

# list_include_item "10 11 12" "2"
function list_include_item {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    result=0
  else
    result=1
  fi
  return $result
}

function CheckConfig(){
	# Check config start
	computing_list="Eggroll Spark STANDALONE"
	spark_federation_list="RabbitMQ Pulsar"
	algorithm_list="Basic NN ALL"
	device_list="CPU IPCL GPU"

	if ! $(list_include_item "$computing_list" "$computing"); then
		echo "[ERROR]: Please check whether computing is one of $computing_list"
		exit 1
	fi

	if [ $computing == "Eggroll" ]; then
		if [ $federation != "OSX" ] ||  [ $storage != "Eggroll" ]; then
			echo "[ERROR]: Please select the correct engine. When eggroll is selected as the computing engine, both Federation and Storage must be osx/eggroll engines!"
			exit 1
		fi
	fi

	if ! $(list_include_item "$algorithm_list" "$algorithm"); then
		echo "[ERROR]: Please check whether algorithm is one of $algorithm_list"
		exit 1
	fi

	if ! $(list_include_item "$device_list" "$device"); then
		echo "[ERROR]: Please check whether algorithm is one of $device_list"
		exit 1
	fi

	echo "Configuration check done!"
	# Check config end
}


GenerateConfig() {
	for ((i = 0; i < ${#party_list[*]}; i++)); do

		eval party_id=\${party_list[${i}]}
		eval party_ip=\${party_ip_list[${i}]}
		eval serving_ip=\${serving_ip_list[${i}]}

		eval venv_dir=/data/projects/python/venv
		eval python_path=${deploy_dir}/python:${deploy_dir}/eggroll/python
		eval data_dir=${deploy_dir}/data-dir

		eval nodemanager_ip=nodemanager
		eval nodemanager_port=4671
		eval nodemanager_port_db=4671

		eval clustermanager_ip=clustermanager
		eval clustermanager_port=4670
		eval clustermanager_port_db=4670

		eval proxy_ip=rollsite
		eval proxy_port=9370

		eval fateboard_ip=fateboard
		eval fateboard_port=8080
		eval fateboard_username="${fateboard_username}"
		eval fateboard_password="${fateboard_password}"

		eval fate_flow_ip=fateflow
		eval fate_flow_grpc_port=9360
		eval fate_flow_http_port=9380
		eval fml_agent_port=8484

		eval db_ip="${mysql_ip}"
		eval db_user="${mysql_user}"
		eval db_password="${mysql_password}"
		eval db_name="${mysql_db}"
		eval db_serverTimezone="${serverTimezone}"

		eval exchange_ip=${exchangeip}

		# gpu_count defaulet 1
		eval gpu_count="${gpu_count:-1}"

		echo package $party_id start!

		rm -rf confs-"$party_id"/
		mkdir -p confs-"$party_id"/confs
		cp -r training_template/public/* confs-"$party_id"/confs/

		# Generate confs packages

		if [ "$computing" == "Eggroll" ]; then
			# if the computing is Eggroll, use eggroll anyway
			cp -r training_template/backends/eggroll confs-$party_id/confs/
			cp training_template/docker-compose-eggroll.yml confs-$party_id/docker-compose.yml

			# eggroll config
			#db connect inf
			# use the fixed db name here
			sed -i "s#<jdbc.url>#jdbc:mysql://${db_ip}:3306/${db_name}?useSSL=false\&serverTimezone=${db_serverTimezone}\&characterEncoding=utf8\&allowPublicKeyRetrieval=true#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<jdbc.username>#${db_user}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<jdbc.password>#${db_password}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties

			#clustermanager & nodemanager
			sed -i "s#<clustermanager.host>#${clustermanager_ip}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<clustermanager.port>#${clustermanager_port}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<nodemanager.host>#${nodemanager_ip}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<nodemanager.port>#${nodemanager_port}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<party.id>#${party_id}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties

			#pythonpath, very import, do not modify."
			sed -i "s#<python.path>#/data/projects/fate/python:/data/projects/fate/eggroll/python#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties

			#javahome
			sed -i "s#<java.home>#/usr/lib/jvm/java-1.8.0-openjdk#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<java.classpath>#conf/:lib/*#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties

			sed -i "s#<rollsite.host>#${proxy_ip}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
			sed -i "s#<rollsite.port>#${proxy_port}#g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
		fi

		cp ${WORKINGDIR}/.env ./confs-$party_id
		echo "NOTEBOOK_HASHED_PASSWORD=${notebook_hashed_password}" >> ./confs-$party_id/.env

		# Modify the configuration file
		
		# federatedai/fateflow-${computing}-${algorithm}-${device}:${version}
		
		# eggroll or spark-worker
		if [ "$computing" == "Eggroll" ]; then
			sed -i "s#image: \"\${FATEFlow_IMAGE}:\${FATEFlow_IMAGE_TAG}\"#image: \"\${FATEFlow_IMAGE}${Suffix}:\${FATEFlow_IMAGE_TAG}\"#g" ./confs-"$party_id"/docker-compose.yml
			sed -i "s#image: \"\${EGGRoll_IMAGE}:\${EGGRoll_IMAGE_TAG}\"#image: \"\${EGGRoll_IMAGE}${Suffix}:\${EGGRoll_IMAGE_TAG}\"#g" ./confs-"$party_id"/docker-compose.yml
		fi

		# GPU
		if [ "$device" == "GPU" ]; then
      line=0 # line refers to the line number of the fateflow `command` line in docker-compose.yaml
      if [ "$computing" == "Eggroll" ]; then
          line=141
      fi
      sed -i "${line}i\\
    deploy:\\
      resources:\\
        reservations:\\
          devices:\\
          - driver: nvidia\\
            count: $gpu_count\\
            capabilities: [gpu]" ./confs-"$party_id"/docker-compose.yml
		fi
		# RegistryURI
		if [ "$RegistryURI" != "" ]; then
		
			if [ "${RegistryURI: -1}" != "/" ]; then
				RegistryURI="${RegistryURI}/"
			fi
			
			sed -i "s#RegistryURI=.*#RegistryURI=${RegistryURI}/#g" ./confs-"$party_id"/.env
		fi

		# replace namenode in training_template/public/fate_flow/conf/service_conf.yaml
		if [ "$name_node" != "" ]; then
			sed -i "s#name_node: hdfs://namenode:9000#name_node: ${name_node}#g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml
		fi

		# update serving ip
		# sed -i "s/fate-serving/${serving_ip}/g" ./confs-"$party_id"/docker-compose.yml

		# update the path of shared_dir
		shared_dir="confs-${party_id}/shared_dir"

		# create directories
		for value in "examples" "fate" "data"; do
			mkdir -p "${shared_dir}"/${value}
		done

		sed -i "s|<path-to-host-dir>|${dir}/${shared_dir}|g" ./confs-"$party_id"/docker-compose.yml

		# Start the general config rendering
		# fateboard
		sed -i "s#^server.port=.*#server.port=${fateboard_port}#g" ./confs-"$party_id"/confs/fate_board/conf/application.properties
    sed -i "s#^fateflow.url=.*#fateflow.url=http://${fate_flow_ip}:${fate_flow_http_port}#g" ./confs-"$party_id"/confs/fate_board/conf/application.properties
    sed -i "s#^server.board.login.username=.*#server.board.login.username=${fateboard_username}#g" ./confs-"$party_id"/confs/fate_board/conf/application.properties
    sed -i "s#^server.board.login.password=.*#server.board.login.password=${fateboard_password}#g" ./confs-"$party_id"/confs/fate_board/conf/application.properties
                
		echo fateboard module of "$party_id" done!

		# mysql
		
		{
			echo "CREATE DATABASE IF NOT EXISTS ${db_name};" 
			echo "CREATE DATABASE IF NOT EXISTS fate_flow;" 
			echo "CREATE USER '${db_user}'@'%' IDENTIFIED BY '${db_password}';"
			echo "GRANT ALL ON *.* TO '${db_user}'@'%';" 
		} >> ./confs-"$party_id"/confs/mysql/init/insert-node.sql

		if [[ "$computing" == "Eggroll" ]]; then
			echo 'USE `'${db_name}'`;' >>./confs-$party_id/confs/mysql/init/insert-node.sql
			echo "show tables;" >>./confs-$party_id/confs/mysql/init/insert-node.sql		
			sed -i "s/eggroll_meta/${db_name}/g" ./confs-$party_id/confs/mysql/init/create-eggroll-meta-tables.sql
		else
			rm -f ./confs-$party_id/confs/mysql/init/create-eggroll-meta-tables.sql
		fi
		echo mysql module of $party_id done!

		# fate_flow
		sed -i "s/party_id: .*/party_id: \"${party_id}\"/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml
		sed -i "s/name: <db_name>/name: '${db_name}'/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml
		sed -i "s/user: <db_user>/user: '${db_user}'/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml
		sed -i "s/passwd: <db_passwd>/passwd: '${db_password}'/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml
		sed -i "s/host: <db_host>/host: '${db_ip}'/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml
		sed -i "s/127.0.0.1:8000/${serving_ip}:8000/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml

		# if [[ "$computing" == "STANDALONE" ]] ; then
		# 	sed -i "s#spark.master .*#spark.master                      local[*]#g" ./confs-$party_id/confs/spark/spark-defaults.conf
		# fi

		# compute_core
		sed -i "s/nodes: .*/nodes: 1/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml
		sed -i "s/cores_per_node: .*/cores_per_node: $compute_core/g" ./confs-$party_id/confs/fate_flow/conf/service_conf.yaml

		if [[ "$computing" == "Eggroll" ]]; then
			sed -i "s/eggroll.session.processors.per.node=.*/eggroll.session.processors.per.node=$compute_core/g" ./confs-$party_id/confs/eggroll/conf/eggroll.properties
		fi
		echo fate_flow module of $party_id done!

		# federation config
		# OSX
		sed -i "s/self.party=9999/self.party=${party_id}/g" ./confs-$party_id/confs/osx/conf/broker.properties
		if [[ "$federation" == "OSX" ]]; then
			cat >./confs-$party_id/confs/osx/conf/route_table.json <<EOF
{
  "route_table":
  {
		$(for ((j = 0; j < ${#party_list[*]}; j++)); do
				if [ "${party_id}" == "${party_list[${j}]}" ]; then
					continue
				fi
				echo "
		\"${party_list[${j}]}\": {
			\"default\": [{
		 		\"ip\": \"${party_ip_list[${j}]}\",
				\"port\": 9370
			    }]
		},
	"
			done)
		"${party_id}": {
			"fateflow": [{
				"ip": "${fate_flow_ip}",
				"port": ${fate_flow_grpc_port}
			}]
		}
  },
	"self_party":[
		"${party_id}"
	],
  "permission":
  {
    "default_allow": true
  }
}
EOF
		fi

		echo proxy module of $party_id done!

		# package of $party_id
		tar -czf ./outputs/confs-$party_id.tar ./confs-$party_id
		rm -rf ./confs-$party_id
		echo package $party_id done!

		if [ "$exchange_ip" != "" ]; then
			# handle exchange
			echo exchange module start!
			module_name=exchange
			cd ${WORKINGDIR}
			rm -rf confs-exchange/
			mkdir -p confs-exchange/conf/eggroll
                        mkdir -p confs-exchange/conf/osx
			cp ${WORKINGDIR}/.env confs-exchange/

			cp training_template/docker-compose-exchange.yml confs-exchange/docker-compose.yml
			cp -r training_template/backends/eggroll/conf/* confs-exchange/conf/eggroll
                        cp -r training_template/backends/osx/conf/* confs-exchange/conf/osx 
			if [ "$RegistryURI" != "" ]; then
				sed -i 's#federatedai#${RegistryURI}/federatedai#g' ./confs-exchange/docker-compose.yml
			fi

			sed -i "s#<rollsite.host>#${proxy_ip}#g" ./confs-exchange/conf/eggroll/eggroll.properties
			sed -i "s#<rollsite.port>#${proxy_port}#g" ./confs-exchange/conf/eggroll/eggroll.properties
			sed -i "s#<party.id>#exchange#g" ./confs-exchange/conf/eggroll/eggroll.properties
			sed -i "s/coordinator=.*/coordinator=exchange/g" ./confs-exchange/conf/eggroll/eggroll.properties
			sed -i "s/ip=.*/ip=0.0.0.0/g" ./confs-exchange/conf/eggroll/eggroll.properties
			cat >./confs-exchange/conf/osx/broker/route_table.json <<EOF
{
    "route_table": {
$(for ((j = 0; j < ${#party_list[*]}; j++)); do
			if [ j < ${#party_list[*]}-1]; then
				echo "        \"${party_list[${j}]}\": {
            \"default\": [
                {
                    \"ip\": \"${party_ip_list[${j}]}\",
                    \"port\": 9370
                }
            ]
        },"
		    else
			    echo "        \"${party_list[${j}]}\": {
            \"default\": [
                {
                    \"ip\": \"${party_ip_list[${j}]}\",
                    \"port\": 9370
                }
            ]
        }"
		    fi
		done)
    },
    "self_party": [
        "default"
    ]
}
EOF
                        #rm -rf ./confs-exchange/conf/osx/broker/route_table.json
                        #cp ./confs-exchange/conf/eggroll/route_table.json ./confs-exchange/conf/osx/broker
			tar -czf ./outputs/confs-exchange.tar ./confs-exchange
			rm -rf ./confs-exchange
			echo exchange module done!
		fi
	done
}

ShowUsage() {
	echo "Usage: "
	echo "Generate configuration: bash generate_config.sh"
}

CleanOutputDir() {
	if [ -d ${WORKINGDIR}/outputs ]; then
		rm -rf ${WORKINGDIR}/outputs
	fi
	mkdir ${WORKINGDIR}/outputs
}

main() {
	if [ "$1" != "" ]; then
		ShowUsage
		exit 1
	else
		CleanOutputDir
		CheckConfig
		GenerateConfig
	fi

	exit 0
}

main $@
