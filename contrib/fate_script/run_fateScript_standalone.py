import os
import sys
import json
import operator
from functools import reduce
from arch.api.utils import file_utils

def get_role_list():
    with open(file_utils.get_project_base_directory() + "/contrib/fate_script/conf/route.json") as fp:
        route_json = json.load(fp)
    print("keys:", route_json.keys())
    role_list = []
    for key in route_json.keys():
        role_list.append(route_json[key]['role'])
    return reduce(operator.add, role_list)


def run_fate_script_standalone(job_id, work_mode):
    role_list = get_role_list()

    for role in role_list:
        runtime_conf_path = file_utils.get_project_base_directory() + "/contrib/fate_script/conf/"+ str(role) + "_runtime_conf.json"
        run_cmd = "python fateScript.py "+ str(role) + " " + str(job_id) + " " + str(runtime_conf_path) + " " + str(work_mode) + "&"
        print("run cmd:",format(run_cmd))
        os.system(run_cmd)
    

if __name__ == "__main__":
    job_id = sys.argv[1]
    work_mode = sys.argv[2]
    run_fate_script_standalone(job_id, work_mode)

