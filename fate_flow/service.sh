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

export PYTHONPATH=
log_dir="${PYTHONPATH}/logs"
venv=
# FATE deployment suggested path
# venv=/data/projects/fate/venv/

module=fate_flow_server.py

getpid() {
    pid=`lsof -i:9380 | awk 'NR==2{print $2}'`
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
    sleep 4
    getpid
    if [[ ${pid} == "" ]]; then
        mklogsdir
        source ${venv}/bin/activate
        nohup python fate_flow_server.py > ${log_dir}/console.log 2>&1 &
        sleep 2
        getpid
        if [[ -n ${pid} ]]; then 
           echo "service start sucessfully. pid: ${pid}"
        else
           echo "service start failed, please check ../logs/console.log"
        fi
    else
        echo "service already started. pid: ${pid}"
    fi
}

stop() {
    getpid
    if [[ -n ${pid} ]]; then
        echo "killing:
        `ps aux | grep ${pid} | grep -v grep`"
        kill -9 ${pid}
        if [[ $? -eq 0 ]]; then
            echo "killed"
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
