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
src_base_path=
dst_base_path=

rsync_args="-avzhK --progress --delete \
    --exclude 'audit' \
    --exclude '*log*' \
    --exclude '*conf*' \
    --exclude '*cert*' \
    --exclude 'data-dir' \
    --exclude '*.tar.gz' \
    --exclude '*.out' \
    --exclude '*DS_Store*' \
    --exclude '*lmdb*' \
    --exclude '*pid.file*' \
    --exclude '*venv*' \
    --exclude 'exchange' \
    --exclude '*standalone*' \
    --exclude '*python_bak*' \
    --exclude '*LMDB*' \
    --exclude '*mdb*' \
    --exclude '*IN_MEMORY*' \
    --exclude '*conf.json' "

#echo "$rsync_args"

deploy() {
    echo "[INFO] deploy"
    echo "================================="
    for target in "${dst[@]}"; do
        echo
        echo "[INFO] processing ${target}"
        echo "------------------"
        
        ssh app@${target}  "rsync ${rsync_args} ${dst_base_path}/fate/ ${dst_base_path}/fate.old/"
        echo ""
        echo ""
        echo "[INFO] backing up for ${target}"
        echo "------------------"
        echo ""
        eval "$backup_cmd"
        echo ""
        echo "[INFO] deploying on ${target}"
        echo "------------------"
        echo ""
        deploy_cmd="rsync "${rsync_args}" ${src_base_path}/fate/ app@${target}:${dst_base_path}/fate/"
        eval "$deploy_cmd"
        echo "------------------"
    done
}

try() {
    echo "[INFO] try (dry-run)"
    echo "================================="
    echo ""
    for target in "${dst[@]}"; do
        echo
        echo "[INFO] trying ${target}"
        echo "------------------"
        cmd="rsync --dry-run "${rsync_args}" ${src_base_path}/fate/ app@${target}:${dst_base_path}/fate/"
        eval "$cmd"
    done
    echo "------------------"
}

rollback() {
    echo "[INFO] rollback"
    echo "================================="

    for target in "${dst[@]}"; do
        echo ""
        echo ""
        echo "[INFO] rolling back ${target}"
        echo "------------------"
        echo ""
        rsync "${rsync_args}" app@${target}:${dst_base_path}/fate.old/ app@${target}:${dst_base_path}/fate/
        echo "------------------"
    done
}

restart() {
    echo "[INFO] restart"
    echo "================================="

    for target in "${dst[@]}"; do
        echo ""
        echo ""
        echo "[INFO] restarting ${target}"
        echo "------------------"
        echo ""
        ssh app@${target} "cd ${dst_base_path}/fate; sh services.sh current restart; exit"
        echo "------------------"
    done
}

overwrite() {
    echo "[INFO] overwrite"
    echo "================================="

    for target in "${dst[@]}"; do
        echo ""
        echo ""
        echo "[INFO] overwriting ${target}"
        echo "------------------"
        echo ""
        cmd="rsync -avzhK --progress --delete --exclude '*venv*' ${src_base_path}/fate/ app@${target}:${dst_base_path}/fate/"
        eval "$cmd"
        echo "------------------"
    done
}

usage() {
    envs_names=${!envs[@]}
    echo "usage: ${script_name} {${envs_names// /|}} {deploy|try|rollback|overwrite|restart}"
}


if [[ "$1" == "usage" ]]; then
    usage
    exit 0
fi


env_str=${envs[$1]}
dst=(${env_str})


if [[ -z ${env_str} ]]; then
    echo "[ERROR] env \"$1\" does not exist."
    usage
    exit -1
fi

echo "[INFO] env: $1. targets:" 
echo "${env_str}"
echo "==============================="    


case "$2" in
    deploy)
        deploy
        ;;
    try)
        try
        ;;
    rollback)
        rollback
        ;;
    overwrite)
        overwrite
        ;;
    restart)
        restart
        ;;
    *)
        usage
        exit -1
esac

echo "[INFO] operation \"$0 $@\" completed"
