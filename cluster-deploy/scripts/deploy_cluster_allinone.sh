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
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source ./default_configurations.sh
source ./allinone_cluster_configurations.sh

service_modules=(mysql redis fate_flow fateboard proxy federation roll metaservice egg)
party_names=(a b)

generate_multinode_configuration() {
    sed -i.bak "s#user=.*#user=${user}#g" ./multinode_cluster_configurations.sh
    sed -i.bak "s#deploy_dir=.*#deploy_dir=${deploy_dir}#g" ./multinode_cluster_configurations.sh
    sed -i.bak "s#party_list=.*#party_list=\(${party_list[*]}\)#g" ./multinode_cluster_configurations.sh
    sed -i.bak "s#db_auth=.*#db_auth=\(${db_auth[*]}\)#g" ./multinode_cluster_configurations.sh
    sed -i.bak "s#redis_password=.*#redis_password=${redis_password}#g" ./multinode_cluster_configurations.sh
    sed -i.bak "s#cxx_compile_flag=.*#cxx_compile_flag=${cxx_compile_flag}#g" ./multinode_cluster_configurations.sh
    deploy_party_names=()
    for ((i=0;i<${#party_list[*]};i++))
    do
	    for service_module in "${service_modules[@]}"
	    do
	        configuration_item=${party_names[i]}_${service_module}
            if [[ "${service_module}" == "egg" ]];then
                sed -i.bak "s#${configuration_item}=.*#${configuration_item}=\(${node_list[i]}\)#g" ./multinode_cluster_configurations.sh
            else
                sed -i.bak "s#${configuration_item}=.*#${configuration_item}=${node_list[i]}#g" ./multinode_cluster_configurations.sh
            fi
	    done
        deploy_party_names[i]=${party_names[i]}
    done
    sed -i.bak "s#party_names=.*#party_names=\(${deploy_party_names[*]}\)#g" ./multinode_cluster_configurations.sh
}

all() {
    init_env
    for ((i=0;i<${#support_modules[*]};i++))
    do
        deploy_modules[i]=${support_modules[i]}
	done
    deploy
}

usage() {
    echo "usage: $0 {binary|build} {all|[module1, ...]}"
}


case "$2" in
    usage)
        usage
        ;;
    *)
        generate_multinode_configuration
        sh ./deploy_cluster_multinode.sh $@
        ;;
esac