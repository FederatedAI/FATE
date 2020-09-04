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
import re
import sys
import json
import time
import socket
import psutil
import datetime
import threading
import argparse
import subprocess
from eggroll.core.session import ErSession
from eggroll.roll_pair.roll_pair import RollPairContext
from eggroll.utils.log_utils import get_logger

L = get_logger()

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-t","--time", type=int, help="Sleep time wait, default value 0s", default=0)
arg_parser.add_argument("-n","--nodes", type=int, help="Eggroll session processors per node, default value 1", default=1)
arg_parser.add_argument("-p","--partitions", type=int, help="Total partitions, default value 1", default=1)
args = arg_parser.parse_args()

def str_generator(include_key=True, row_limit=10, key_suffix_size=0, value_suffix_size=0):
    for i in range(row_limit):
        if include_key:
            yield str(i) + "s"*key_suffix_size, str(i) + "s"*value_suffix_size
        else:
            yield str(i) + "s"*value_suffix_size

def round2(x):
    return str(round(x / 1024 / 1024 / 1024, 2))

def print_red(str):
    print("\033[1;31;40m\t" + str + "\033[0m")

def print_green(str):
    print("\033[1;32;40m\t" + str + "\033[0m")

def print_yellow(str):
    print("\033[1;33;40m\t" + str + "\033[0m")

def check_actual_max_threads():
    def getMemInfo(fn):
        def query_cmd(cmd):
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0].decode().strip().split('\n')
            return p[0]
 
        def get_host_ip():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
            return ip
        fate_flow_client = "/data/projects/fate/python/fate_flow/fate_flow_client.py"
        mem_info = {}
        mem_info["Ip"] = get_host_ip()
        mem_info["route_table"] = query_cmd("if [ -f $EGGROLL_HOME/conf/route_table.json ];then array=(`cat $EGGROLL_HOME/conf/route_table.json |grep -E 'ip|port'`); echo ${array[@]}; else echo 0; fi")
        mem_info["data_access"] = query_cmd("ps aux |grep data_access_server |grep -v grep |wc -l")
        mem_info["data_test"] = query_cmd("curl -X POST --header 'Content-Type: application/json' -d '{\"local\": {\"role\": \"host\", \"party_id\": 10000}, \"id_type\":\"phone\", \"encrypt_type\":\"md5\"}' 'http://127.0.0.1:9350/v1/data/query_imported_id_library_info'")
        mem_info["directory"] = query_cmd("if [ -d /data/projects/fdn/FDN-DataAcces ];then echo 1; else echo 0; fi")
        mem_info["services"] = ['ClusterManagerBootstrap','NodeManagerBootstrap','rollsite','fate_flow_server.py','fateboard','mysql']
        mem_info["job_run"] = query_cmd("if [ -f %s ];then python %s -f query_job -s running | grep f_job_id |wc -l; else echo -1; fi" %(fate_flow_client,fate_flow_client))
        mem_info["job_wait"] = query_cmd("if [ -f %s ];then python %s -f query_job -s waiting | grep f_job_id |wc -l; else echo -1; fi" %(fate_flow_client,fate_flow_client))
        mem_info["job_thread"] = []
        mem_info["jobs"] = query_cmd("array=(`python %s -f query_job -s running | grep f_job_id |awk -F: '{print $2}' |awk -F '\"' '{print $2}'`);echo ${array[@]}" %(fate_flow_client))
        mem_info["job_mem"] = []
        mem_info["data_num"] = mem_info["data_test"].split(':')[-1].split('}')[0]
        for job_id in mem_info["jobs"]:
            mem_info["job_thread"] = query_cmd("ps -ef |grep egg_pair |grep -v grep |grep %s |wc -l" %(job_id))
            mem_info["job_mem"] = query_cmd("ps aux |grep egg_pair |grep %s |awk '{sum+=$6};END {print sum}'" %(job_id))
        mem_info["server_mem"] = {}
        mem_info["thread"] = {}
        for service in mem_info["services"]:
            mem_info["thread"][service] = query_cmd("ps -ef |grep %s |grep -v grep |wc -l" %(service))
            mem_info["server_mem"][service] = str(query_cmd("ps aux |grep %s |grep -v grep |awk '{sum+=$6};END {print sum}'" %(service)))
        return mem_info

    session = ErSession(options={"eggroll.session.processors.per.node": args.nodes})
    try:
        ctx = RollPairContext(session)
        rp = ctx.parallelize(str_generator(row_limit=1000), options={'total_partitions': args.partitions})
        result = rp.with_stores(func=getMemInfo)
        print_green(str(datetime.datetime.now()))
        for node in result:
            print_green("==============This is node " + str(node[0]) + ":" + node[1]["Ip"] + "===========================================")
            print_green("-------------default route check-------------------------------------------------------")
            if node[1]["route_table"] == 0:
                print_red("[ERROR] eggroll route is not configured, please check data/projects/fate/eggroll/conf/route_table.json file if it is existed!")
            else:
                print_green("[OK] eggroll route configured!")
                print_green(node[1]["route_table"])

            print_green("--------------data_access service check-------------------------------------------------")
            if int(node[1]["data_access"]) == 0:
                if int(node[1]["directory"]) == 0:
                    print_red("[ERROR] data_access service and directory not found, please check if it is installed!")
                else:
                    print_yellow("[WARNING] data_access not running or check /data/projects/fdn/FDN-DataAcces directory") 
            else:
                if int(node[1]["data_num"]) == 0 or int(node[1]["data_num"]) == 201:
                    print_green(node[1]["data_test"])
                    print_green("[OK] Installed and running data_access service!")
                else:
                    print_yellow(node[1]["data_test"])
                    print_yellow("[WARNING] data_access service not available, please check host and host route!")

            print_green("--------------fate service check-------------------------------------------------------")
            for server in node[1]["services"]:
                if int(node[1]["thread"][server]) > 0:
                    print_green("[OK] the " + server.ljust(23) + " service is running , number of processes is : " + str(node[1]["thread"][server]) + "; used memory : " + str(node[1]["server_mem"][server]) + "KB.")
                else:
                    print_yellow("[WARNING] the " + server + " service not running, please check service status.")

            print_green("--------------fate_flow jobs process and mem info check--------------------------------------------------")
            if int(node[1]["job_run"]) == -1:
                print_red("[ERROR] There is no such fate_flow_client.py file, please check fate_flow server if it is running!")
            else:
                print_green("[OK] Number of tasks running is " + node[1]["job_run"])
                print_green("[OK] Number of tasks waiting is " + node[1]["job_wait"])
                if int(node[1]["job_run"]) > 0:
                    for job_id in node[1]["jobs"].split(" "):
                        print_green("[OK] running task job_id : " + job_id + " run " + str(node[1]["job_thread"]) + " processes; used memory : " + str(node[1]["job_mem"]) + "KB.")

            print("\n")
    finally:
        session.kill()


if __name__ == '__main__':
    if args.time == 0:
        check_actual_max_threads()
    else:
        while 1:
            check_actual_max_threads()
            time.sleep(args.time)
s()
    else:
        while 1:
            check_actual_max_threads()
            time.sleep(args.time)
