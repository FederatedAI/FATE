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
import os
import time
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-t","--time", type=int, help="Sleep time wait, default value 0s", default=0)
args = arg_parser.parse_args()

if args.time == 0:
    os.system('sh ./cluster_env_check.sh')
else:
    while 1:
        os.system('sh ./cluster_env_check.sh')
        time.sleep(args.time)
