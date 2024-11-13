#!/bin/bash

# Copyright 2019-2022 VMware, Inc.
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
cd $BASEDIR
WORKINGDIR=$(pwd)

# fetch fate-python image
source ${WORKINGDIR}/.env
source ${WORKINGDIR}/parties.conf

cd ${WORKINGDIR}

Deploy() {
	if [ "$1" = "" ]; then
		echo "No party id was provided, please check your arguments "
		exit 1
	fi

	while [ "$1" != "" ]; do
		case $1 in
		splitting_proxy)
			shift
			DeployPartyInternal $@
			break
			;;
		all)
			for party in ${party_list[*]}; do
				if [ "$2" != "" ]; then
					case $2 in
					--training)
						DeployPartyInternal $party
						if [ "${exchangeip}" != "" ]; then
							DeployPartyInternal exchange
						fi
						;;
					--serving)
						DeployPartyServing $party
						;;
					esac
				else
					DeployPartyInternal $party
					DeployPartyServing $party
					if [ "${exchangeip}" != "" ]; then
						DeployPartyInternal exchange
					fi
				fi
			done
			break
			;;
		*)
			if [ "$2" != "" ]; then
				case $2 in
				--training)
					DeployPartyInternal $1
					break
					;;
				--serving)
					DeployPartyServing $1
					break
					;;
				esac
			else
				DeployPartyInternal $1
				DeployPartyServing $1
			fi
			;;
		esac
		shift

	done
}

Delete() {
	if [ "$1" = "" ]; then
		echo "No party id was provided, please check your arguments "
		exit 1
	fi

	while [ "$1" != "" ]; do
		case $1 in
		all)
			for party in ${party_list[*]}; do
				if [ "$2" != "" ]; then
					DeleteCluster $party $2
				else
					DeleteCluster $party
				fi
			done
			if [ "${exchangeip}" != "" ]; then
				DeleteCluster exchange
			fi
			break
			;;
		*)
			DeleteCluster $@
			break
			;;
		esac
	done
}

DeployPartyInternal() {
	target_party_id=$1
	# should not use localhost at any case
	target_party_ip="127.0.0.1"

	# check configuration files
	if [ ! -d ${WORKINGDIR}/outputs ]; then
		echo "Unable to find outputs dir, please generate config files first."
		return 1
	fi
	if [ ! -f ${WORKINGDIR}/outputs/confs-${target_party_id}.tar ]; then
		echo "Unable to find deployment file of training for party $target_party_id, please generate it first."
		return 0
	fi
	# extract the ip address of the target party
	if [ "$target_party_id" = "exchange" ]; then
		target_party_ip=${exchangeip}
	elif [ "$2" != "" ]; then
		target_party_ip="$2"
	else
		for ((i = 0; i < ${#party_list[*]}; i++)); do
			if [ "${party_list[$i]}" = "$target_party_id" ]; then
				target_party_ip=${party_ip_list[$i]}
			fi
		done
	fi
	# verify the target_party_ip
	if [ "$target_party_ip" = "127.0.0.1" ]; then
		echo "Unable to find Party: $target_party_id, please check you input."
		return 1
	fi

	if [ "$3" != "" ]; then
		user=$3
	fi

	handleLocally confs
	if [ "$local_flag" == "true" ]; then
		return 0
	fi

	scp -P ${SSH_PORT} ${WORKINGDIR}/outputs/confs-$target_party_id.tar $user@$target_party_ip:~/
	#rm -f ${WORKINGDIR}/outputs/confs-$target_party_id.tar
	echo "$target_party_ip training cluster copy is ok!"
	ssh -p ${SSH_PORT} -tt $user@$target_party_ip <<eeooff
mkdir -p $dir
rm -f $dir/confs-$target_party_id.tar
mv ~/confs-$target_party_id.tar $dir
cd $dir
tar -xzf confs-$target_party_id.tar
cd confs-$target_party_id
docker compose down
docker volume rm -f confs-${target_party_id}_shared_dir_examples
docker volume rm -f confs-${target_party_id}_shared_dir_fate
docker volume rm -f confs-${target_party_id}_sdownload_dir
docker volume rm -f confs-${target_party_id}_fate_flow_logs

docker compose up -d
cd ../
rm -f confs-${target_party_id}.tar
exit
eeooff
	echo "party ${target_party_id} deploy is ok!"
}

DeployPartyServing() {
	target_party_id=$1
	# should not use localhost at any case
	target_party_serving_ip="127.0.0.1"

	# check configuration files
	if [ ! -d ${WORKINGDIR}/outputs ]; then
		echo "Unable to find outputs dir, please generate config files first."
		return 1
	fi
	if [ ! -f ${WORKINGDIR}/outputs/serving-${target_party_id}.tar ]; then
		echo "Unable to find deployment file of serving for party $target_party_id, please generate it first."
		return 0
	fi
	# extract the ip address of the target party
	for ((i = 0; i < ${#party_list[*]}; i++)); do
		if [ "${party_list[$i]}" = "$target_party_id" ]; then
			target_party_serving_ip=${serving_ip_list[$i]}
		fi
	done
	# verify the target_party_ip
	if [ "$target_party_serving_ip" = "127.0.0.1" ]; then
		echo "Unable to find Party : $target_party_id serving address, please check you input."
		return 1
	fi

	handleLocally serving
	if [ $local_flag == "true" ]; then
		return
	fi

	scp -P ${SSH_PORT} ${WORKINGDIR}/outputs/serving-$target_party_id.tar $user@$target_party_serving_ip:~/
	echo "party $target_party_id serving cluster copy is ok!"
	ssh -p ${SSH_PORT} -tt $user@$target_party_serving_ip <<eeooff
mkdir -p $dir
rm -f $dir/serving-$target_party_id.tar
mv ~/serving-$target_party_id.tar $dir
cd $dir
tar -xzf serving-$target_party_id.tar
cd serving-$target_party_id
docker compose down
docker compose up -d
cd ../
rm -f serving-$target_party_id.tar
exit
eeooff
	echo "party $target_party_id serving cluster deploy is ok!"
}

DeleteCluster() {
	target_party_id=$1
	cluster_type=$2
	target_party_serving_ip="127.0.0.1"
	target_party_ip="127.0.0.1"

	# extract the ip address of the target party
	if [ "$target_party_id" == "exchange" ]; then
		target_party_ip=${exchangeip}
	else
		for ((i = 0; i < ${#party_list[*]}; i++)); do
			if [ "${party_list[$i]}" = "$target_party_id" ]; then
				target_party_ip=${party_ip_list[$i]}
			fi
		done
	fi
	
	# echo "target_party_ip: $target_party_ip"

	for ((i = 0; i < ${#party_list[*]}; i++)); do
		if [ "${party_list[$i]}" = "$target_party_id" ]; then
			target_party_serving_ip=${serving_ip_list[$i]}
		fi
	done

	#	echo "target_party_ip: $target_party_ip"
	#	echo "cluster_type: $cluster_type"

	# delete training cluster
	if [ "$cluster_type" == "--training" ]; then
		ssh -p ${SSH_PORT} -tt $user@$target_party_ip <<eeooff
cd $dir/confs-$target_party_id
docker compose down
exit
eeooff
		echo "party $target_party_id training cluster is deleted!"
	# delete serving cluster
	elif [ "$cluster_type" == "--serving" ]; then
		ssh -p ${SSH_PORT} -tt $user@$target_party_serving_ip <<eeooff
cd $dir/serving-$target_party_id
docker compose down
exit
eeooff
		echo "party $target_party_id serving cluster is deleted!"
	# delete training cluster and serving cluster
	else
		# if party is exchange then delete exchange cluster
		if [ "$target_party_id" == "exchange" ]; then
			ssh -p ${SSH_PORT} -tt $user@$target_party_ip <<eeooff
cd $dir/confs-$target_party_id
docker compose down
exit
eeooff
		else
			if [ "$target_party_serving_ip" != "" ]; then
			ssh -p ${SSH_PORT} -tt $user@$target_party_serving_ip <<eeooff
cd $dir/serving-$target_party_id
docker compose down
exit
eeooff
			fi
			if [ "$target_party_ip" != "" ]; then
			ssh -p ${SSH_PORT} -tt $user@$target_party_ip <<eeooff
cd $dir/confs-$target_party_id
docker compose down
exit
eeooff
			fi
			echo "party $target_party_id training cluster is deleted!"
			echo "party $target_party_id serving cluster is deleted!"
		fi
	fi
}

ShowUsage() {
	echo "Usage: "
	echo "Deploy all parties or specified partie(s): bash docker_deploy.sh partyid1[partyid2...] | all"
}

handleLocally() {
	type=$1
	for ip in $(hostname -I); do
		if [ "$target_party_ip" == "$ip" ]; then
			mkdir -p $dir
			tar -xf ${WORKINGDIR}/outputs/${type}-${target_party_id}.tar -C $dir
			cd ${dir}/${type}-${target_party_id}
			docker compose down
			docker compose up -d
			local_flag="true"
			return 0
		fi
	done
	local_flag="false"
}

main() {

	if [ "$1" = "" ] || [ "$1" = "--help" ]; then
		ShowUsage
		exit 1
	elif [ "$1" = "--delete" ] || [ "$1" = "--del" ]; then
		shift
		Delete $@
	else
		Deploy "$@"
	fi

	for ((i = 0; i < ${#party_list[*]}; i++)); do
	  if [ $party_list[$i] != "exchange" ]; then
        echo "
   Use  ${party_ip_list[$i]}:8080 to access fateboard of party: ${party_list[$i]}
   Use  ${party_ip_list[$i]}:20000 to access notebook of party: ${party_list[$i]}"
      fi
      if [[ "$computing" == "spark"* ]]; then
        echo "   Use  ${party_ip_list[$i]}:8888 to access Spark of party: ${party_list[$i]}"
      fi
      if [ ${serving_ip_list[$i]} ]; then
        echo "   Use  ${party_ip_list[$i]}:8350 to access serving-admin of party: ${party_list[$i]}"
      fi
	done

	exit 0
}

main $@
