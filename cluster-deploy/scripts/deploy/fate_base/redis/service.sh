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
set -e
basepath=$(cd `dirname $0`;pwd)
user=`whoami`

getpid() {
    echo $(ps -aux | grep redis-server | grep ${basepath} | grep -v grep | awk '{print $2}') > redis_pid
}

status() {
    getpid
    pid=`cat redis_pid`
    if [[ -n ${pid} ]]; then
        echo "status:`ps aux | grep ${pid} | grep redis-server | grep ${basepath} | grep -v grep`"
    else
        echo "service not running"
    fi
}

start() {
    getpid
    pid=`cat redis_pid`
    if [[ ${pid} == "" ]]; then
        nohup $basepath/bin/redis-server redis.conf &
        if [[ $? -eq 0 ]]; then
            sleep 2
            getpid
            pid=`cat redis_pid`
            echo "service start sucessfully. pid: ${pid}"
        else
            echo "service start failed"
        fi
    else
        echo "service already started. pid: ${pid}"
    fi
}

stop() {
    getpid
    pid=`cat redis_pid`
    if [[ -n ${pid} ]]; then
        echo "killing:`ps aux | grep ${pid} | grep redis-server | grep ${basepath} | grep -v grep`"
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