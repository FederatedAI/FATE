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
error_exit (){
    echo "ERROR: $1 !!"
    exit 1
}

[ ! -e "$JAVA_HOME/bin/java" ] && JAVA_HOME=$HOME/jdk/java
[ ! -e "$JAVA_HOME/bin/java" ] && JAVA_HOME=/usr/java
[ ! -e "$JAVA_HOME/bin/java" ] && error_exit "Please set the JAVA_HOME variable in your environment, We need java(x64)!"
export JAVA_HOME
export JAVA="$JAVA_HOME/bin/java"
export BASE_DIR=$(dirname $0)/..
export CLASSPATH=.:${BASE_DIR}/conf:${BASE_DIR}/lib/*:${CLASSPATH}

#===========================================================================================
# JVM Configuration
#===========================================================================================
# The RAMDisk initializing size in MB on Darwin OS for gc-log
DIR_SIZE_IN_MB=600

choose_gc_log_directory()
{
    case "`uname`" in
        Darwin)
            if [ ! -d "/Volumes/RAMDisk" ]; then
                # create ram disk on Darwin systems as gc-log directory
                DEV=`hdiutil attach -nomount ram://$((2 * 1024 * DIR_SIZE_IN_MB))` > /dev/null
                diskutil eraseVolume HFS+ RAMDisk ${DEV} > /dev/null
                echo "Create RAMDisk /Volumes/RAMDisk for gc logging on Darwin OS."
            fi
            GC_LOG_DIR="/Volumes/RAMDisk"
        ;;
        *)
            # check if /dev/shm exists on other systems
            if [ -d "/dev/shm" ]; then
                GC_LOG_DIR="/dev/shm"
            else
                GC_LOG_DIR=${BASE_DIR}
            fi
        ;;
    esac
}

choose_gc_options()
{
    JAVA_MAJOR_VERSION=$("$JAVA" -version 2>&1 | head -1 | cut -d'"' -f2 | sed 's/^1\.//' | cut -d'.' -f1)
    if [ -z "$JAVA_MAJOR_VERSION" ] || [ "$JAVA_MAJOR_VERSION" -lt "8" ] ; then
      JAVA_OPT="${JAVA_OPT} -XX:+UseConcMarkSweepGC -XX:+UseCMSCompactAtFullCollection -XX:CMSInitiatingOccupancyFraction=70 -XX:+CMSParallelRemarkEnabled -XX:SoftRefLRUPolicyMSPerMB=0 -XX:+CMSClassUnloadingEnabled -XX:SurvivorRatio=8 -XX:-UseParNewGC"
    else
      JAVA_OPT="${JAVA_OPT} -XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:G1ReservePercent=25 -XX:InitiatingHeapOccupancyPercent=30 -XX:SoftRefLRUPolicyMSPerMB=0"
    fi

    if [ -z "$JAVA_MAJOR_VERSION" ] || [ "$JAVA_MAJOR_VERSION" -lt "9" ] ; then
      JAVA_OPT="${JAVA_OPT} -verbose:gc -Xloggc:${GC_LOG_DIR}/rmq_srv_gc_%p_%t.log -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintGCApplicationStoppedTime -XX:+PrintAdaptiveSizePolicy"
      JAVA_OPT="${JAVA_OPT} -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=5 -XX:GCLogFileSize=30m"
    else
      JAVA_OPT="${JAVA_OPT} -XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:G1ReservePercent=25 -XX:InitiatingHeapOccupancyPercent=30 -XX:SoftRefLRUPolicyMSPerMB=0"
      JAVA_OPT="${JAVA_OPT} -Xlog:gc*:file=${GC_LOG_DIR}/rmq_srv_gc_%p_%t.log:time,tags:filecount=5,filesize=30M"
    fi
}

choose_gc_log_directory

JAVA_OPT="${JAVA_OPT} -server -Xms2g -Xmx2g"
choose_gc_options
JAVA_OPT="${JAVA_OPT} -XX:-OmitStackTraceInFastThrow"
JAVA_OPT="${JAVA_OPT} -XX:+AlwaysPreTouch"
JAVA_OPT="${JAVA_OPT} -XX:MaxDirectMemorySize=15g"
JAVA_OPT="${JAVA_OPT} -XX:-UseLargePages -XX:-UseBiasedLocking"
JAVA_OPT="${JAVA_OPT} ${JAVA_OPT_EXT}"

set -e
getpid() {
  if [ -e "./bin/broker.pid" ]; then
    pid=$(cat ./bin/broker.pid)
  fi
  if [[ -n ${pid} ]]; then
    count=$(ps -ef | grep $pid | grep -v "grep" | wc -l)
    if [[ ${count} -eq 0 ]]; then
      rm ./bin/broker.pid
      unset pid
    fi
  fi

}

mklogsdir() {
  if [[ ! -d "logs" ]]; then
    mkdir logs
  fi
}

start() {
  echo "try to start $1"
  module=broker
  main_class=com.osx.broker.Bootstrap
  getpid $module
  if [[ ! -n ${pid} ]]; then   JAVA_OPT="${JAVA_OPT}  "
    mklogsdir
    JAVA_OPT="${JAVA_OPT} -cp conf/broker/:lib/*:extension/*:${BASE_DIR}/${project_name}-${module}-${module_version}.jar"
    JAVA_OPT="${JAVA_OPT} ${main_class}"
    JAVA_OPT="${JAVA_OPT} -c ${configpath} "
    echo $JAVA ${JAVA_OPT}
    nohup  $JAVA ${JAVA_OPT} >/dev/null 2>&1 &
    inspect_pid 5 $!
    if [[ "$exist" = 1 ]]; then
       echo $! >./bin/${module}.pid
       getpid ${module}
       echo "service start sucessfully. pid: ${pid}"
    else
       echo "service start failed, "
    fi
  else
    echo "service already started. pid: ${pid}"
  fi
}

debug() {
  echo "try to start $1"
  module=broker
  main_class=com.osx.broker.Bootstrap
  getpid $module
  if [[ ! -n ${pid} ]]; then   JAVA_OPT="${JAVA_OPT}  "
    mklogsdir
    JAVA_OPT="${JAVA_OPT} -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=8008 -cp conf/broker/:lib/*:extension/*:${BASE_DIR}/${project_name}-${module}-${module_version}.jar"
    JAVA_OPT="${JAVA_OPT} ${main_class}"
    JAVA_OPT="${JAVA_OPT} -c ${configpath} "
    echo $JAVA ${JAVA_OPT}
    nohup  $JAVA ${JAVA_OPT} >/dev/null 2>&1 &
    inspect_pid 5 $!
    if [[ "$exist" = 1 ]]; then
       echo $! >./bin/${module}.pid
       getpid ${module}
       echo "service start sucessfully. pid: ${pid}"
    else
       echo "service start failed, "
    fi
  else
    echo "service already started. pid: ${pid}"
  fi
}


status() {
  getpid $1
  if [[ -n ${pid} ]]; then
    echo "status: $(ps -f -p ${pid})"
    exit 0
  else
    echo "service not running"
    exit 1
  fi
}

stop() {
  getpid $1
  if [[ -n ${pid} ]]; then
    echo "killing: $(ps -p ${pid})"
    kill ${pid}
    if [[ $? -eq 0 ]]; then
      #此函数检查进程，判断进程是否存在
      echo "please wait"
      inspect_pid 5 ${pid}
      if [[ "$exist" = 0 ]]; then
        echo "killed"
      else
        echo "please retry"
      fi
    else
      echo "kill error"
    fi
  else
    echo "service not running"
  fi
}

inspect_pid() {
  total=0
  exist=0
  if [[ -n $2 ]]; then
    while [[ $total -le $1 ]]
    do
      count=$(ps -ef | grep $2 | grep -v "grep" | wc -l)
      total=$(($total+1))
      if [[ ${count} -ne 0 ]]; then
        sleep 1
        exist=1
       else
        exist=0
        return
       fi
    done
  fi
}
