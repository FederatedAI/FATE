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

cd $(dirname "$0")
cur_dir=$(pwd)
first_test=1
failed_count=0

run_test() {
    file=$1
    echo "start to run test "$file
    if [ $first_test == 1 ]; then
		coverage run $file 2>test.log
	else
		coverage run -a $file 2>test.log
	fi

	failed_single_test=$(cat test.log | grep FAILED | wc -l)
	failed_count=$(($failed_count+$failed_single_test))
	echo $failed_count
	cat test.log
	first_test=0
}

traverse_folder() {
    for file in $(ls ${1});
    do
        file_fullname=$1/$file
        if [ -d $file_fullname ]; then
            traverse_folder $file_fullname
		elif [[ $file =~ _test.py$ ]] && [[ $1 =~ /test$ || $1 =~ tests$ ]]; then
            if [[ $file_fullname =~ "ftl" ]]; then
                continue
            else
                run_test $file_fullname
            fi
        fi
    done
}

traverse_folder $cur_dir/..
# traverse_folder $cur_dir/../../fate_flow/tests/api_tests

echo "there are "$failed_count" failed test"

if [ $failed_count -gt 0 ]; then
	exit 1
fi

