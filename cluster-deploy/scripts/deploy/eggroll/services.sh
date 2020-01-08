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

eval action=\$$#
installdir=
export JAVA_HOME=
export PATH=$JAVA_HOME/bin:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
export PYTHONPATH=
modules=(meta-service egg roll storage-service-cxx)

if ! test -e $installdir/logs/storage-service-cxx;then
	mkdir -p $installdir/logs/storage-service-cxx
fi

main() {
	case "$module" in
		egg)
			main_class=com.webank.ai.eggroll.framework.egg.Egg
			export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=''
			;;
		roll)
			main_class=com.webank.ai.eggroll.framework.Roll
			;;
		meta-service)
			main_class=com.webank.ai.eggroll.framework.MetaService
			;;
		storage-service-cxx)
                        main_class=storage-service
			target=storage-service
			port=7778
			dirdata=$installdir/data-dir
			export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:$installdir/$module/third_party/lib
			export GLOG_log_dir=$installdir/logs/storage-service-cxx
			;;
		*)
			echo "usage: $module {meta-service|egg|roll}"
			exit -1
	esac
}

action() {
	case "$action" in
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
			echo "usage: $action {start|stop|status|restart}"
			exit -1
	esac
}

all() {
	for module in "${modules[@]}"; do
		main
        echo
        echo "--------------"
		echo "[INFO] $module:${main_class}"
        echo "[INFO] processing: ${module} ${action}"
        action
        echo "--------------"
	done
}

usage() {
    echo "usage: $0 {all|[module1, ...]} {start|stop|status|restart}"
}

multiple() {
    total=$#
    action=${!total}
    for (( i=1; i<total; i++)); do
        module=${!i//\//}
		main
        echo
        echo "--------------"
		echo "[INFO] $module:${main_class}"
        echo "[INFO] processing: ${module} ${action}"
        action
        echo "--------------"
    done
}

getpid() {
     echo $(ps aux | grep ${module} | grep ${main_class} | grep ${installdir} | grep -v grep | grep -v $0 | awk '{print $2}') > ${module}/${module}_pid
}

status() {
    getpid
    pid=`cat ${module}/${module}_pid`
    if [[ -n ${pid} ]]; then
        echo "status:
        `ps aux | grep ${pid} | grep ${module} | grep ${main_class} |grep ${installdir} | grep -v grep`"
        return 0
    else
        echo "service not running"
        return 1
    fi
}

start() {
    getpid
    pid=`cat ${module}/${module}_pid`
    if [[ ${pid} == "" ]]; then
		if [[ "$module" == "storage-service-cxx" ]]; then
			${installdir}/$module/${target} -p $port -d ${dirdata} >/dev/null 2>/dev/null &
			echo $!>${module}/${module}_pid
                else
			java -cp "$installdir/${module}/conf/:$installdir/${module}/lib/*:$installdir/${module}/eggroll-${module}.jar" ${main_class} -c $installdir/${module}/conf/${module}.properties >/dev/null 2>/dev/null &
			echo $!>${module}/${module}_pid
		fi
		sleep 2
		getpid
                pid=`cat ${module}/${module}_pid`
		if [[ -n ${pid}  ]]; then
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
    pid=`cat ${module}/${module}_pid`
    if [[ -n ${pid} ]]; then
        echo "killing:
        `ps aux | grep ${pid} | grep ${installdir} | grep ${main_class} |grep ${module} | grep -v grep`"
        kill -9 ${pid}
        if [[ $? -eq 0 ]]; then
            echo "" > ${module}/${module}_pid
            echo "killed"
        else
            echo "kill error"
        fi
    else
        echo "service not running"
    fi
}

case "$1" in
    all)
        all $@
        ;;
    usage)
        usage
        ;;
    *)
        multiple $@
        ;;
esac


