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
source ./check_iplist.sh

cd $EGGROLL_HOME

echo "----------------------$EGGROLL_HOME/conf/eggroll.properties--------------------"
cat $EGGROLL_HOME/conf/eggroll.properties | grep -v ^# | grep -v ^$
echo ""
echo "-----------------------$EGGROLL_HOME/conf/route_table.json---------------------"
cat $EGGROLL_HOME/conf/route_table.json | grep -v ^# | grep -v ^$

for ip in ${iplist[@]};do
    echo "------------------diff $ip with ./conf/eggroll.properties-------------------------"
    ssh $user@$ip "cat $EGGROLL_HOME/conf/eggroll.properties" | diff - conf/eggroll.properties
	echo ""
done

cd $cwd
