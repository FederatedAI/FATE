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
source ./bin/common.sh
#export JAVA_HOME=/data/projects/fate/common/jdk/jdk-8u192
#export PATH=$PATH:$JAVA_HOME/bin

basepath=$(cd `dirname $0`;pwd)
configpath=$(cd $basepath/conf;pwd)
libpath=$(cd $basepath/lib;pwd)
#module=transfer
#main_class=com.firework.transfer.Bootstrap
module_version=1.0.0-alpha
project_name=osx



case "$1" in
    start)
        start $2
        status $2
        ;;
    debug)
        debug $2
        status $2
        ;;
    stop)
        stop $2
        ;;
    status)
        status $2
        ;;
    restart)
        stop $2
        sleep 0.5
        start  $2
        status $2
        ;;
    rebudeg)
        stop $2
        sleep 0.5
        debug  $2
        status $2
        ;;
    *)
        echo "usage: $0 {start|stop|status|restart}"
        exit 1

esac
