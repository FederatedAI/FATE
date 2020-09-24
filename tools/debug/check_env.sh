#!/bin/bash
#  Copyright (c) 2019 - now, Eggroll Authors. All Rights Reserved.
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
#
cwd=$(cd `dirname $0`; pwd)

get_property() {
    property_value=`grep $1 $2 | cut -d '=' -f 2-`
    test_value $1 $2 ${property_value}
}

echo_red() {
    echo -e "\e[1;31m $1\e[0m"
}

echo_green() {
    echo -e "\e[1;32m $1\e[0m"
}

echo_yellow() {
    echo -e "\e[1;33m $1\e[0m"
}

check_max_count() {
    value=`cat $1`
    if [ $value -ge 65535 ];then
        echo_green "[OK] $1 is ok."
    else
        echo_red "[ERROR] please check $1, no less than 65535."
    fi  
}

check_file_count() {
    value=`cat $1 | grep $2 | awk '{print $4}'`
    for v in ${value[@]};do
        test_value $1 $2 $v
    done 
}

test_value() {
    if [ $3 -ge 65535 ];then
        echo_green "[OK] $1 in $2 is ok."
    else
        echo_red "[ERROR] please check $1 in $2, no less than 65535."
    fi
}

echo_green `date +"%Y-%m-%d_%H:%M:%S"`

echo_green "=============check max user processes============"
check_max_count "/proc/sys/kernel/threads-max"
get_property "kernel.pid_max" "/etc/sysctl.conf"
check_max_count "/proc/sys/kernel/pid_max"
check_max_count "/proc/sys/vm/max_map_count"

echo_green "=============check max files count=============="
check_file_count "/etc/security/limits.conf" "nofile"
check_file_count "/etc/security/limits.d/80-nofile.conf" "nofile"
get_property "fs.file-max" "/etc/sysctl.conf"
check_max_count "/proc/sys/fs/file-max"

mem_total=`free -m | grep Mem | awk '{print $2}' | tr -cd "[0-9,.]"`
mem_used=`free -m | grep Mem | awk '{print $3}' | tr -cd "[0-9],."`
swap_total=`free -m | grep Swap | awk '{print $2}' | tr -cd "[0-9,.]"`
swap_used=`free -m | grep Swap | awk '{print $3}' | tr -cd "[0-9,.]"`

echo_green "=============Memory used and total==============="
echo_yellow "[WARNING] MemTotal:`awk 'BEGIN{printf "%.2f%%\n",('$mem_total'/1024)}'`G, MemUsed:`awk 'BEGIN{printf "%.2f%%\n",('$mem_used'/1024)}'`G, MemUsed%:`awk 'BEGIN{printf "%.2f%%\n",('$mem_used'/'$mem_total')*100}'`"
echo_green "=============SwapMem used and total==============="
echo_yellow "[WARNING] SwapTotal:`awk 'BEGIN{printf "%.2f%%\n",('$swap_total'/1024)}'`G, SwapUsed:`awk 'BEGIN{printf "%.2f%%\n",('$swap_used'/1024)}'`G, SwapUsed%:`awk 'BEGIN{printf "%.2f%%\n",('$swap_used'/'$swap_total')*100}'`"
echo_green "=============Disk use and total=================="
echo_yellow "[WARNING] `df -lh | grep /data`"


