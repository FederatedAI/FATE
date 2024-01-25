#!/bin/bash

#
#  Copyright 2019 The osx Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# --------------- Color Definitions ---------------
esc_c="\033[0m"
error_c="\033[31m"
ok_c="\033[32m"

project_name=osx
module=broker
module_version=1.0.0
main_class=org.fedai.osx.broker.Bootstrap

DIR_SIZE_IN_MB=600
#highlight_c="\033[43m"

# --------------- Logging Functions ---------------
print_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local overwrite=$2

    # Check if we need to overwrite the current line
    if [ "$overwrite" == "overwrite" ]; then
        echo -ne "\r${ok_c}[${timestamp}][INFO]${esc_c} $1"
    else
        echo -e "${ok_c}[${timestamp}][INFO]${esc_c} $1"
    fi
}
print_ok() {
    local overwrite=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [ "$overwrite" == "overwrite" ]; then
        echo -ne "\r${ok_c}[${timestamp}][ OK ]${esc_c} $1"
    else
        echo -e "${ok_c}[${timestamp}][ OK ]${esc_c} $1"
    fi
}
print_error() {
    local overwrite=$3
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [ "$overwrite" == "overwrite" ]; then
        echo -ne "\r${error_c}[${timestamp}][ ER ]${esc_c} $1: $2"
    else
        echo -e "${error_c}[${timestamp}][ ER ]${esc_c} $1: $2"
    fi
}

cwd=$(cd `dirname $0`; pwd)
cd $cwd
export OSX_HOME=`pwd`
osx_conf=${OSX_HOME}/conf

cd ${OSX_HOME}
print_info "OSX_HOME=${OSX_HOME}"

eval action=\$$#
start_mode=1

if [ $action = starting ];then
	action=start
	start_mode=0
elif [ $action = restarting ];then
	action=restart
	start_mode=0
fi



# Get the PID of the process
getpid() {
  pid=`ps aux | grep ${osx_conf} | grep ${main_class} | grep -v grep | awk '{print $2}'`
	if [[ -n ${pid} ]]; then
		return 1
	else
		return 0
	fi
}

# Get the PID of the process using a specific port
get_port_pid() {
  pid=$(lsof -i:${port} | grep 'LISTEN' | awk 'NR==1 {print $2}')
  if [[ -n ${pid} ]]; then
  	return 0
  else
  	return 1
  fi
}

# --------------- Functions for stop---------------
# Function to kill a process
kill_process() {
    local pid=$1
    local signal=$2
    kill ${signal} "${pid}" 2>/dev/null
}

# --------------- Functions for info---------------
# Print usage information for the script
usage() {

	    echo -e "${ok_c}osx${esc_c}"
      echo "------------------------------------"
      echo -e "${ok_c}Usage:${esc_c}"
      echo -e "  `basename ${0}` start          - Start the server application."
      echo -e "  `basename ${0}` stop           - Stop the server application."
      echo -e "  `basename ${0}` shut           - Force kill the server application."
      echo -e "  `basename ${0}` status         - Check and report the status of the server application."
      echo -e "  `basename ${0}` restart [time] - Restart the server application. Optionally, specify a sleep time (in seconds) between stop and start."
      echo -e "  `basename ${0}` debug          - Start the server application in debug mode."
      echo ""
      echo -e "${ok_c}Examples:${esc_c}"
      echo "  `basename ${0}` start"
      echo "  `basename ${0}` restart 5"
      echo ""
      echo -e "${ok_c}Notes:${esc_c}"
      echo "  - The restart command, if given an optional sleep time, will wait for the specified number of seconds between stopping and starting the service."
      echo "    If not provided, it defaults to 2 seconds."
      echo "  - Ensure that the required Java environment is correctly configured on the system."
      echo ""
      echo "For more detailed information, refer to the script's documentation or visit the official documentation website."
}

main() {
		print_info "${project_name}:${main_class}"
		print_info "Processing: ${project_name} ${action}"
		action "$@"
}


mklogsdir() {
	if [[ ! -d "${OSX_HOME}/logs" ]]; then
		mkdir -p ${OSX_HOME}/logs
	fi
}

# --------------- Functions for status---------------
# Check the status of the service
status() {
  print_info "---------------------------------status---------------------------------"
    getpid
	# check service is up and running
	if [[ -n ${pid} ]]; then
    print_ok "Check service ${project_name} is started: PID=${pid}${esc_c}"
    print_info "The service status is:
    `ps aux | grep ${pid} | grep ${main_class} | grep -v grep`"
    return 0
	else
		print_error "The ${project_name} service is not running"
		return 1
	fi
}
# check java environment
check_java_environment() {
    #检查是否已经设置 JAVA_HOME 环境变量
    if [ -n "$JAVA_HOME" ]; then
        print_ok "JAVA_HOME is set to $JAVA_HOME"
        export JAVA="$JAVA_HOME/bin/java"
    else
        print_error "JAVA_HOME is not set"
        export JAVA="java"
        exit 1
    fi
    #检查 Java 可执行文件是否在系统 PATH 中
    if command -v java &> /dev/null; then
        print_ok "Java is installed and available in the system PATH"
    else
        print_error "Java is not found in the system PATH"
        exit 1
    fi

    #检查 Java 版本
    java_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
    if [ -n "$java_version" ]; then
        print_ok "Java version is $java_version"
    else
        print_error "Java version information is not available"
        exit 1
    fi

}

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
                GC_LOG_DIR=${OSX_HOME}
            fi
        ;;
    esac
}

choose_gc_options()
{

    JAVA_MAJOR_VERSION=$($JAVA -version 2>&1 | head -1 | cut -d'"' -f2 | sed 's/^1\.//' | cut -d'.' -f1)
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

    JAVA_OPT="${JAVA_OPT} -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=${BASE_DIR}/oom/heapdump.hprof "
}

# Start service
start() {
  print_info "--------------------------------starting--------------------------------"
  print_info "Checking Java environment..."
  # check the java environment
  check_java_environment
  getpid
	if [[ $? != 1 ]]; then
	  choose_gc_log_directory
    choose_gc_options

    JAVA_OPT="${JAVA_OPT} -server -Xms4g -Xmx4g"
    JAVA_OPT="${JAVA_OPT} -XX:-OmitStackTraceInFastThrow"
    JAVA_OPT="${JAVA_OPT} -XX:+AlwaysPreTouch"
    JAVA_OPT="${JAVA_OPT} -XX:MaxDirectMemorySize=15g"
    JAVA_OPT="${JAVA_OPT} -XX:-UseLargePages -XX:-UseBiasedLocking"
    JAVA_OPT="${JAVA_OPT} ${JAVA_OPT_EXT}"
		mklogsdir
    JAVA_OPT="${JAVA_OPT} -cp conf/broker/:lib/*:extension/*:${OSX_HOME}/lib/${project_name}-${module}-${module_version}.jar"
    properties_file="conf/broker/broker.properties"
    property_name="eggroll.version"
    eggroll_version=$(grep -w "^$property_name" "$properties_file" | cut -d'=' -f2)
    eggroll_version=$(echo $eggroll_version | sed -e 's/^[[:space:]]*//')
    if [[ $eggroll_version == 2* ]]; then
      JAVA_OPT="${JAVA_OPT}:pb_lib/osx-pb-v2*.jar"
    else
      JAVA_OPT="${JAVA_OPT}:pb_lib/osx-pb-v3*.jar"
    fi
    JAVA_OPT="${JAVA_OPT} ${main_class}"
    JAVA_OPT="${JAVA_OPT} -c ${osx_conf} "
    cmd="$JAVA ${JAVA_OPT}"
		print_info "The command is: ${cmd}"

		if [ $start_mode = 0 ];then
			exec $cmd >>/dev/null 2>&1
		else
			exec $cmd >>/dev/null 2>&1 &
		fi
    # wait for connect DB
    print_info "Waiting for start service..."
    sleep 5
    getpid
		if [[ $? -eq 1 ]]; then
      print_ok "The ${project_name} service start sucessfully. PID=${pid}"
		else
			print_error "The ${project_name} service start failed"
		fi
	else
		print_info "The ${project_name} service already started. PID=${pid}"
	fi
}

debug() {
    print_info "--------------------------------starting--------------------------------"
  print_info "Checking Java environment..."
  # check the java environment
  check_java_environment
  getpid
	if [[ $? != 1 ]]; then
	  choose_gc_log_directory
    choose_gc_options
    JAVA_OPT="${JAVA_OPT} -server -Xdebug -Xrunjdwp:transport=dt_socket,server=y,suspend=n,address=7007  -Xms4g -Xmx4g"
    JAVA_OPT="${JAVA_OPT} -XX:-OmitStackTraceInFastThrow"
    JAVA_OPT="${JAVA_OPT} -XX:+AlwaysPreTouch"
    JAVA_OPT="${JAVA_OPT} -XX:MaxDirectMemorySize=15g"
    JAVA_OPT="${JAVA_OPT} -XX:-UseLargePages -XX:-UseBiasedLocking"
    JAVA_OPT="${JAVA_OPT} ${JAVA_OPT_EXT}"
		mklogsdir
    JAVA_OPT="${JAVA_OPT} -cp conf/broker/:lib/*:extension/*:${OSX_HOME}/lib/${project_name}-${module}-${module_version}.jar"
    JAVA_OPT="${JAVA_OPT} ${main_class}"
    JAVA_OPT="${JAVA_OPT} -c ${osx_conf} "
    cmd="$JAVA ${JAVA_OPT}"
		print_info "The command is: ${cmd}"

		if [ $start_mode = 0 ];then
			exec $cmd >>/dev/null 2>&1
		else
			exec $cmd >>/dev/null 2>&1 &
		fi
    # wait for connect DB
    print_info "Waiting for start service..."
    sleep 5
    getpid
		if [[ $? -eq 0 ]]; then
      print_ok "The ${project_name} service debug sucessfully. PID=${pid}"
		else
			print_error "The ${project_name} service debug failed"
		fi
	else
		print_info "The ${project_name} service already started. PID=${pid}"
	fi
}

# --------------- Functions for stop---------------
# Stop service
stop() {
  print_info "--------------------------------stopping--------------------------------"
  getpid
	if [[ -n ${pid} ]]; then
		print_info "The system is stopping the ${project_name} service. PID=${pid}"
		print_info "The more information:`ps aux | grep ${pid} | grep ${main_class} | grep -v grep`"
	  for _ in {1..100}; do
        sleep 0.1
        kill_process "${pid}"
        getpid
        if [ -z "${pid}" ]; then
            print_ok "Stop ${project_name} success "
            return
        fi
    done
    kill_process "${pid}" -9 && print_ok "Stop the service ${project_name} success (SIGKILL)" || print_error "Stop service failed"
	else
		print_ok "The ${project_name} service is not running(NOT ACTIVE))"
	fi
}
# Shut service(FORCE KILL). now not use, stop has force kill
shut() {
  print_info "--------------------------------shutting--------------------------------"
  getpid
	if [[ -n ${pid} ]]; then
	  print_info "The ${project_name} service is force killing. PID=${pid}"
		print_info "The more information:
		`ps aux | grep ${pid} | grep ${main_class} | grep -v grep`"
		kill -9 ${pid}
		sleep 1
		flag=0
		while [ $flag -eq 0 ]
		do
			getpid
			flag=$?
		done
		print_info "The ${project_name} service is force kill success"
	else
		print_info "The ${project_name} service is not running"
	fi
}

action() {
	case "$action" in
	  debug)
	  stop
	  sleep_time=${3:-2}
    print_info "Waiting ${sleep_time} seconds"
    sleep "$sleep_time"
	  debug
	  status
	  ;;
		start)
			start
			status
			;;
		stop)
			stop
			;;
#		kill)
#			shut
#			status
#			;;
		status)
			status
			;;
		restart)
			stop
			sleep_time=${3:-2}  # 默认 sleep_time 为 5，如果传入了参数，则使用传入的值
			print_info "Waiting ${sleep_time} seconds"
      sleep "$sleep_time"
			start
			status
			;;
		*)
			usage
			exit 1
	esac
}


# --------------- Main---------------
# Main case for control
case "$1" in
	start|restart|debug|starting|restarting|stop|status)
		main "$@"
		;;
	*)
	  usage
    exit 1
		;;
esac

cd $cwd

