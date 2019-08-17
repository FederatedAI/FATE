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

modules=(federation meta-service egg roll proxy storage-service-cxx)

cwd=`pwd`

all() {
    echo "[INFO] processing all modules"
    echo "=================="
    echo
    for module in "${modules[@]}"; do
	echo
        echo "[INFO] processing: ${module}"
        echo "--------------"
        cd ${module}
        bash service.sh $2
        cd ${cwd}
        echo "=================="
    done
}

current() {
    echo "[INFO] processing current active modules"
    echo "=================="
    
    action=${@: -1}

    for module in "${modules[@]}"; do
        echo
        echo $module
        echo "----------------"
        cd ${module}
        echo ""
        `bash service.sh status >> /dev/null`
	state=$?
        if [[ ${state} -eq 1 ]]; then
            echo "[INFO] processing ${module} ${action}"
            bash service.sh ${action}
        else
            echo "[INFO] ${module} not running"
        fi
        cd ${cwd}
    done
}

usage() {
    echo "usage: $0 {all|current|[module1, ...]} {start|stop|status|restart}"
}

multiple() {
    total=$#
    action=${!total}
    for (( i=1; i<total; i++)); do
        module=${!i}
        echo
        echo "[INFO] processing: ${module} ${action}"
        echo "=================="
        cd ${module}
        bash service.sh ${action}
        cd -
        echo "--------------"
    done
}

case "$1" in
    all)
        all $@
        ;;
    current)
        current $@
        ;;
    usage)
        usage
        ;;
    *)
        multiple $@
        ;;
esac

cd ${cwd}
echo "=================="
echo "process finished"
