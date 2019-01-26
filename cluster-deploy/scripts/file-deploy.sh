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

declare -A envs

envs[cluster1]="
127.0.0.1
"

envs[cluster2]="
192.168.0.1
"

script_name=$0
args=" -avzhK --progress --delete "

excludes=" --exclude '*nohup.out' --exclude '*src/data/*' --exclude '*server_conf.json' "

usage() {
    envs_names=${!envs[@]}
    echo "usage: ${script_name} {${envs_names// /|}|<ip_address>} local_path remote_path"
}

process() {
    env_str=${envs[$1]}
    targets=(${env_str})

    if [[ -z ${env_str} ]]; then
        echo "[WARN] env \"$1\" does not exist. trying to use [$1] as ip"
        targets=($1)
    fi

    echo "[INFO] env: $1. targets:" 
    echo "${env_str}"
    echo "==============================="    
    for target in "${targets[@]}"; do
        echo
        echo "[INFO] processing ${target}"
        echo "==============================="    

        final_args="${args} ${excludes} $3 app@${target}:$4"

        case $2 in
            act)
                cmd="rsync ${final_args}"
                ;;
            try)
                cmd="rsync --dry-run ${final_args}"
                ;;
            *)
                usage
                exit 1
                ;;
        esac
        echo "${cmd}"
        
        eval ${cmd}
        echo "-------------------"
    done
}

if [[ "$1" == "usage" ]]; then
    usage
    exit 0
fi

process $1 try $2 $3

echo
echo "[WARN] files marked 'delete' above will be deleted"
read -r -p "Do you want to actually perform rsync? [y/n]: " answer

case ${answer} in
    [yY][eE][sS]|[yY])
        echo "Yes"
        countdown=3
        while [[ ${countdown} -ge 0 ]]; do
            echo -ne "[INFO] rsync starts in $countdown s... \033[0K\r"
            sleep 1s
            : $((countdown--))
        done

        process $1 act $2 $3

        echo
        echo "[INFO] local path [$2] has been rsync-ed to [$1]'s remote path [$3]"
        ;;
    [nN][oO]|[nN])
        echo "No"
        ;;
    [aA][bB][oO][rR][tT]|[aA])
        echo "Aborted"
        ;;
    *)
        echo "Invalid input ... exiting"
        exit -1
        ;;
esac

