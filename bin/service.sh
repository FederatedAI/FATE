#!/bin/bash
#
#  Copyright 2019 The fate Authors. All Rights Reserved.
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

# -----------------global params--------------
cwd=$(cd `dirname $0`; pwd)
fate_home=/data/projects/fate
eggroll_modules=(nodemanager clustermanager  dashboard)
fate_modules=(fate-flow fate-board osx eggroll mysql)
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

source_env() {
  echo "================>>>source_env<<<==============="
}

check_python_env(){
  echo "================>>>check_python_env<<<==============="
}

check_python_env(){
  echo "================>>>check_python_env<<<==============="
}

print_split(){
  name=$1
  padding_length=$((20 - ${#name}))
  padded_string=$(printf "%s%*s" "$1" $padding_length " ")
  print_info " -----------------------------------------------------"
  print_info "|component :    ${padded_string}                  |"
  print_info " -----------------------------------------------------"
}

print_end(){
  echo ""
}

eggroll(){
  print_split $1
  cd ${fate_home}/eggroll || exit 1
	case "$1" in
	eggroll)
	  case "$2" in
	    status)
		    ;;
      *)
        "`pwd`/bin/eggroll.sh" all $2
        ;;
    esac

		declare -A array
    array[0]=" ---------------------------"
		pid=`ps aux | grep "${fate_home}/eggroll/lib" | grep "org.fedai.eggroll.clustermanager.Bootstrap" |  grep -v grep | awk '{print $2}'`
    if [[ -n ${pid} ]]; then
      array[1]="|  clustermanager running   |"
	  else
	  	array[1]="|  clustermanager failed    |"
	  fi
    unset pid

    pid=`ps aux | grep "${fate_home}/eggroll/lib" | grep "org.fedai.eggroll.nodemanager.Bootstrap" |  grep -v grep | awk '{print $2}'`
    if [[ -n ${pid} ]]; then
      array[2]="|  nodemanager    running   |"
	  else
	  	array[2]="|  nodemanager    failed    |"
	  fi
    unset pid

    pid=`ps aux | grep "${fate_home}/eggroll/lib" | grep "org.fedai.eggroll.webapp.JettyServer" |  grep -v grep | awk '{print $2}'`
    if [[ -n ${pid} ]]; then
      array[3]="|  dashboard      running   |"
	  else
	  	array[3]="|  dashboard      failed    |"
	  fi
    unset pid

    array[4]=" ---------------------------"
	  for msg in "${array[@]}"; do
      echo "$msg"
    done
		;;
  *)
	  "`pwd`/bin/eggroll.sh" $1 $2
		;;
  esac
	cd "${fate_home}" || exit 1
	print_end
}

osx(){
  print_split $1
  cd ${fate_home}/osx || exit 1
   "`pwd`/service.sh" $2
  cd "${fate_home}" || exit 1
  print_end
}

flow(){
  print_split $1
  cd ${fate_home}/fate_flow || exit 1
   "`pwd`/bin/service.sh" $2
  cd "${fate_home}" || exit 1
  print_end
}

mysql(){
  print_split $1
  cd ${fate_home}/common/mysql/mysql-8.0.28 || exit 1
   "`pwd`/service.sh" $2
  cd "${fate_home}" || exit 1
  print_end
}

board(){
  print_split $1
  cd ${fate_home}/fateboard || exit 1
   "`pwd`/service.sh" $2
  cd "${fate_home}" || exit 1
  print_end
}

all(){
  declare -A array
  array[0]=" ---------------------------"
  if [[ $2 != "status" ]]; then
    mysql mysql $2
    flow fate-flow  $2
    board fate-board  $2
    osx osx  $2
    eggroll eggroll  $2
	fi

  pid=`ps aux | grep "${fate_home}/common/mysql/mysql-8.0.28/bin/mysqld_safe" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[1]="|  mysql          running   |"
	else
		array[1]="|  mysql          failed    |"
	fi
  unset pid


	pid=`ps aux | grep "${fate_home}/fate_flow/python/fate_flow/fate_flow_server.py" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[2]="|  fate-flow      running   |"
	else
		array[2]="|  fate-flow      failed    |"
	fi
  unset pid


	pid=`ps aux | grep "${fate_home}/fateboard/fateboard-" | grep "org.fedai.fate.board.bootstrap.Bootstrap" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[3]="|  fate-board     running   |"
	else
		array[3]="|  fate-board     failed    |"
	fi
  unset pid


	pid=`ps aux | grep "${fate_home}/osx/conf" | grep "org.fedai.osx.broker.Bootstrap" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[4]="|  osx            running   |"
	else
		array[4]="|  osx            failed    |"
	fi
  unset pid


	pid=`ps aux | grep "${fate_home}/eggroll/lib" | grep "org.fedai.eggroll.clustermanager.Bootstrap" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[5]="|  clustermanager running   |"
	else
		array[5]="|  clustermanager failed    |"
	fi
  unset pid

  pid=`ps aux | grep "${fate_home}/eggroll/lib" | grep "org.fedai.eggroll.clustermanager.Bootstrap" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[5]="|  clustermanager running   |"
	else
		array[5]="|  clustermanager failed    |"
	fi
  unset pid

  pid=`ps aux | grep "${fate_home}/eggroll/lib" | grep "org.fedai.eggroll.nodemanager.Bootstrap" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[6]="|  nodemanager    running   |"
	else
		array[6]="|  nodemanager    failed    |"
	fi
  unset pid

  pid=`ps aux | grep "${fate_home}/eggroll/lib" | grep "org.fedai.eggroll.webapp.JettyServer" |  grep -v grep | awk '{print $2}'`
  if [[ -n ${pid} ]]; then
    array[7]="|  dashboard      running   |"
	else
		array[7]="|  dashboard      failed    |"
	fi
  unset pid

  array[8]=" ---------------------------"
	for msg in "${array[@]}"; do
    echo "$msg"
  done
}

status(){
  echo ""
}

# --------------- Functions for info---------------
# Print usage information for the script
usage() {

	    echo -e "${ok_c}FATE${esc_c}"
      echo "------------------------------------"
      echo -e "${ok_c}Usage:${esc_c}"
      echo -e "  `basename ${0}` [component] start          - Start the server application."
      echo -e "  `basename ${0}` [component] stop           - Stop the server application."
      echo -e "  `basename ${0}` [component] status         - Check and report the status of the server application."
      echo -e "  `basename ${0}` [component] restart [time] - Restart the server application. Optionally, specify a sleep time (in seconds) between stop and start."
      echo -e "  The ${ok_c}component${esc_c} include: {fate-flow | fate-board | osx | eggroll |clustermanager | nodemanager | dashboard | mysql | all} "
      echo ""
      echo -e "${ok_c}Examples:${esc_c}"
      echo "  `basename ${0}` fate-flow stop"
      echo "  `basename ${0}` eggroll restart"
}

dispatch(){
    case "$1" in
	all)
		all "$@"
		;;
	clustermanager|nodemanager|dashboard|eggroll)
		eggroll "$@"
		;;
  osx)
    osx "$@"
    ;;
  fate-flow)
    flow "$@"
    ;;
  fate-board)
    board "$@"
    ;;
  mysql)
    mysql "$@"
    ;;
	*)
	  usage
    exit 1
		;;
esac
}


# main func
# --------------- Main---------------
# Main case for control
dispatch "$@"
