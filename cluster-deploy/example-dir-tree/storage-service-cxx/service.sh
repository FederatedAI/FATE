#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export GLOG_log_dir=`pwd`/logs

module=storage-service
target=storage-service

getpid() {
    pid=`ps aux | grep ${target} | grep -v grep | awk '{print $2}'`

    if [[ -n ${pid} ]]; then
        return 1
    else
        return 0
    fi
}

status() {
    getpid
    if [[ -n ${pid} ]]; then
        echo "status:
        `ps aux | grep ${pid} | grep -v grep`"
        exit 1
    else
        echo "service not running"
        exit 0
    fi
}

start() {
    getpid
    if [[ $? -eq 0 ]]; then
        ./${target} -p 7778 -d /data/projects/fate/data-dir >> logs/console.log 2>>logs/error.log &
        if [[ $? -eq 0 ]]; then
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
