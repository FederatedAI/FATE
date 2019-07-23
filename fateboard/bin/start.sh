#!/bin/bash
basepath=$(cd `dirname $0`;pwd)
export  JAVA_HOME=/data/projects/common/jdk/jdk1.8.0_192
configpath=$(cd $basepath/../config;pwd)
routerpath=$(cd $basepath/../ssh;pwd)
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
start(){

    checkpid
    if [ $psid -ne 0 ]; then
        echo "already started pid=$psid"
        else
         nohup $JAVA_HOME/bin/java   -Dspring.config.location=$configpath/application.properties  -Dssh_config_file=$routerpath  -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError  -jar $basepath/../fateboard-0.0.1-SNAPSHOT.jar >/dev/null 2>&1 &
         checkpid
         if [ $psid -ne 0 ];
         then  
             echo "(pid=$psid)[ok]"
         else
             echo "failed"
             exit 1
         fi    
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
start

