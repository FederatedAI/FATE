#!/usr/bin/env bash

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

work_mode=$1
jobid=$2
guest_partyid=$3
host_partyid=$4
if [[ $work_mode -eq 1 ]]; then
    role=$5
fi

cur_dir=$(pwd)
data_dir=$cur_dir/../data
conf_dir=$cur_dir/conf
guest_runtime_conf=$conf_dir/guest_runtime_conf.json
host_runtime_conf=$conf_dir/host_runtime_conf.json


toy() {
    role=$1
    runtime_conf=''
    if [ $role = 'guest' ]; then
        runtime_conf=$guest_runtime_conf
    else
        runtime_conf=$host_runtime_conf
    fi

    cur_runtime_conf=${runtime_conf}_$jobid
    cp $runtime_conf $cur_runtime_conf

    echo "current runtime conf is "$cur_runtime_conf
    echo "training table is"$train_table
    echo $predict_table
    sed -i "s/_work_mode/$work_mode/g" $cur_runtime_conf
    sed -i "s/_guest_party_id/$guest_partyid/g" $cur_runtime_conf
    sed -i "s/_host_party_id/$host_partyid/g" $cur_runtime_conf
    sed -i "s/_jobid/$jobid/g" $cur_runtime_conf

    if [ $role == 'guest' ]; then
        echo "enter guest"
        bash run_guest.sh $cur_runtime_conf $jobid
    else
        echo "enter host"
        nohup bash run_host.sh $cur_runtime_conf $jobid > nohup.host &
    fi

}

if [ $work_mode -eq 0 ]; then
    toy host
    toy guest
elif [[ $role == 'guest' ]]; then
    toy guest
elif [[ $role == 'host' ]]; then
    toy host

fi
