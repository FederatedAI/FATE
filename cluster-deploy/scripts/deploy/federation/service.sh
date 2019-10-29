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

export JAVA_HOME=
export PATH=$JAVA_HOME/bin:$PATH

module=federation
main_class=com.webank.ai.fate.driver.Federation

getpid() {
    pid=`ps aux | grep ${main_class} | grep -v grep | awk '{print $2}'`

    if [[ -n ${pid} ]]; then
        return 1
    else
        return 0
    fi
}

mklogsdir() {
    if [[ ! -d "logs" ]]; then
        mkdir logs
    fi
}

status() {
    getpid
    if [[ -n ${pid} ]]; then
        echo "status:
        `ps aux | grep ${pid} | grep -v grep`"
        return 1
    else
        echo "service not running"
        return 0
    fi
}

start() {
    getpid
    if [[ $? -eq 0 ]]; then
        mklogsdir
        java -cp "conf/:lib/*:fate-${module}.jar" ${main_class} -c conf/${module}.properties >> logs/console.log 2>>logs/error.log &
        if [[ $? -eq 0 ]]; then
            sleep 2
            getpid
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
