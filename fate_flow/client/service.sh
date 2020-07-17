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

FATE_PYTHON_ROOT=$(dirname $(dirname $(readlink -f "$0")))
EGGROLL_HOME=$(dirname ${FATE_PYTHON_ROOT})/eggroll
PYTHON_PATH=${FATE_PYTHON_ROOT}:${EGGROLL_HOME}/python
export PYTHONPATH=${PYTHON_PATH}
export EGGROLL_HOME=${EGGROLL_HOME}
echo "PYTHONPATH: "${PYTHONPATH}
echo "EGGROLL_HOME: "${EGGROLL_HOME}
log_dir=${FATE_PYTHON_ROOT}/logs
venv=/data/projects/fate/common/python/venv

module=fate_flow_server.py

getpid() {
    pid1=`lsof -i:9380 | grep 'LISTEN' | awk '{print $2}'`
    pid2=`lsof -i:9360 | grep 'LISTEN' | awk '{print $2}'`
    if [[ -n ${pid1} && "x"${pid1} = "x"${pid2} ]];then
        pid=$pid1
    elif [[ -z ${pid1} && -z ${pid2} ]];then
        pid=
    fi
}

mklogsdir() {
    if [[ ! -d $log_dir ]]; then
        mkdir -p $log_dir
    fi
}

status() {
    getpid
    if [[ -n ${pid} ]]; then
        echo "status:`ps aux | grep ${pid} | grep -v grep`"
    else
        echo "service not running"
    fi
}

start() {
    getpid
    if [[ ${pid} == "" ]]; then
        mklogsdir
        source ${venv}/bin/activate
        nohup python ${FATE_PYTHON_ROOT}/fate_flow/fate_flow_server.py >> "${log_dir}/console.log" 2>>"${log_dir}/error.log" &
        for((i=1;i<=100;i++));
        do
            sleep 0.1
            getpid
            if [[ -n ${pid} ]]; then
                echo "service start sucessfully. pid: ${pid}"
                return
            fi
        done
        if [[ -n ${pid} ]]; then
           echo "service start sucessfully. pid: ${pid}"
        else
           echo "service start failed, please check ${log_dir}/error.log and ${log_dir}/console.log"
        fi
    else
        echo "service already started. pid: ${pid}"
    fi
}

stop() {
    getpid
    if [[ -n ${pid} ]]; then
        echo "killing: `ps aux | grep ${pid} | grep -v grep`"
        for((i=1;i<=100;i++));
        do
            sleep 0.1
            kill ${pid}
            getpid
            if [[ ! -n ${pid} ]]; then
                echo "killed by SIGTERM"
                return
            fi
        done
        kill -9 ${pid}
        if [[ $? -eq 0 ]]; then
            echo "killed by SIGKILL"
        else
            echo "kill error"
        fi
    else
        echo "service not running"
    fi
}


case "$1" in
    start)
        start
        status
        ;;

    stop)
        stop
        ;;
    status)
        status
        ;;

    restart)
        stop
        start
        status
        ;;
    *)
        echo "usage: $0 {start|stop|status|restart}"
        exit -1
esac
