#!/bin/bash
psid=0
checkpid(){
    JPID=$(ps -ef|grep java|grep fateboard-0.0.1-SNAPSHOT.jar|grep -v grep|awk '{print $2}')
    if [ -z "$JPID" ];
    then
        psid=0
    else
        psid=$JPID
    fi
}
stop(){
    checkpid
    if [ $psid -ne 0 ]; then
        echo -n "Stopping $psid"
        kill -9 $psid
        if [ $? -eq 0 ];then
            echo "stop ok"
        else
            echo "stop failed"
        fi
    fi
}

stop

