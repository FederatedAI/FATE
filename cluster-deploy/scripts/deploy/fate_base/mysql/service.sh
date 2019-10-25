#!/bin/bash
set -e
basepath=$(cd `dirname $0`;pwd)
user=`whoami`

getpid() {
    pid=`ps -ef | grep mysqld_safe | grep -v grep | awk '{print $2}'`
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
        nohup $basepath/bin/mysqld_safe --defaults-file=$basepath/conf/my.cnf --user=$user &
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
        echo "killing:`ps aux | grep ${pid} | grep -v grep`"
        kill -9 ${pid}
        kill -9 `lsof -i:3306 | grep -i "LISTEN" | awk '{print $2}'`
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