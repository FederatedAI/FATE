import argparse
import json
import os
import pprint
import random
import time
from flow_sdk.client import FlowClient

home_dir = os.path.split(os.path.realpath(__file__))[0]
dsl_path = os.path.join(home_dir, "toy_example_dsl.json")
conf_v1_path = os.path.join(home_dir, "toy_example_conf_v1.json")
conf_v2_path = os.path.join(home_dir, "toy_example_conf_v2.json")

flow_client = None

guest_party_id = -1
host_party_id = -1

work_mode = 0
backend = 0
dsl_version = 1

user_name = ""

component_name = 'secure_add_example_0'

GUEST = 'guest'
HOST = 'host'

MAX_TIME = 100
WATI_LOG_TIME = 3


def get_timeid():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path():
    config_dir = os.path.join(home_dir, "test")
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)

    return os.path.join(config_dir, "toy_example_conf.json_" + get_timeid())


def create_new_runtime_config():
    conf_dict = {}
    conf_path = conf_v1_path if dsl_version == 1 else conf_v2_path
    with open(conf_path, "r") as fin:
        conf_dict = json.loads(fin.read())

    if not conf_dict:
        if not os.path.isfile(conf_dict):
            raise ValueError("config file {} dose not exist, please check!".format(conf_path))

        raise ValueError("json format error of toy runtime conf")

    conf_dict["initiator"]["party_id"] = guest_party_id
    conf_dict["role"]["guest"] = [guest_party_id]
    conf_dict["role"]["host"] = [host_party_id]
    if dsl_version == 1:
        conf_dict["job_parameters"]["work_mode"] = work_mode
        conf_dict["job_parameters"]["backend"] = backend
    else:
        conf_dict["job_parameters"]["common"]["work_mode"] = work_mode
        conf_dict["job_parameters"]["common"]["backend"] = backend
        conf_dict["job_parameters"]["role"] = {
            "guest": {"0": {"user": user_name}},
            "host": {"0": {"user": user_name}},
        }

    return conf_dict


def exec_task(dsl_path, config_data):
    dsl_dict = {}
    if dsl_path:
        with open(dsl_path, "r") as fin:
            dsl_dict = json.loads(fin.read())

    result = flow_client.job.submit(config_data=config_data, dsl_data=dsl_dict)
    pprint.pprint (result["data"])
    try:
        status = result["retcode"]
    except:
        raise ValueError("failed to exec task, msg is {}".format(result))

    if status != 0:
        raise ValueError(
            "failed to exec task, status_code:{}, msg is {}".format(status, result))

    job_id = result["jobId"]
    return job_id


def get_job_status(job_id):
    result = flow_client.job.query(job_id=job_id, role="guest", party_id=guest_party_id)
    try:
        retcode = result["retcode"]
    except:
        return "query job status failed"

    if retcode != 0:
        return "query job status failed"

    status = result["data"][0]["f_status"]
    return status


def show_log(job_id, log_level):
    log_dir = os.path.join(home_dir, "test", "log")
    log_path = os.path.join(log_dir, "job_" + job_id + "_log", "guest", str(guest_party_id), "secure_add_example_0")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    result = flow_client.job.log(job_id=job_id, output_path=log_dir)

    if 'retcode' not in result or result['retcode'] != 0:
        raise ValueError("Can not download logs from fate_flow server, error msg is {}".format(stdout))

    if log_level == "error":
        error_log = os.path.join(log_path, "ERROR.log")
        
        with open(error_log, "r") as fin:
            for line in fin:
                print (line.strip())
    else:
        info_log = os.path.join(log_path, "INFO.log")
        with open(info_log, "r") as fin:
            for line in fin:
                if line.find("secure_add_guest") != -1:
                    print (line.strip())


def exec_toy_example(runtime_config):
    jobid = exec_task(dsl_path, runtime_config)

    # print ("toy example is running, jobid is {}".format(jobid))

    for i in range(MAX_TIME):
        time.sleep(1)

        status = get_job_status(jobid)

        if status == "failed":
            show_log(jobid, "error")
            return
        elif status == "success":
            show_log(jobid, "info")
            return
        else:
            print ("job status is {}".format(status))

    raise ValueError("job running time exceed, please check federation or eggroll log")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("guest_party_id", type=int, help="please input guest party id")
    arg_parser.add_argument("host_party_id", type=int, help="please input host party id")
    arg_parser.add_argument("work_mode", type=int,
                            help="please input work_mode, 0 stands for standalone, 1  stands for cluster")
    arg_parser.add_argument("flow_server_ip", type=str, help="please input flow server'ip")
    arg_parser.add_argument("flow_server_port", type=int, help="please input flow server port")
    arg_parser.add_argument("-b", "--backend", type=int, default=0,
                            help="please input backend, 0 stands for eggroll, 1 stands for spark")
    arg_parser.add_argument("-v", "--dsl_version", type=int, default=1,
                            help="please input dsl version to use")
    arg_parser.add_argument("-u", "--user_name", type=str, help="please input user name")

    args = arg_parser.parse_args()

    guest_party_id = args.guest_party_id
    host_party_id = args.host_party_id
    work_mode = args.work_mode
    backend = args.backend
    dsl_version = args.dsl_version
    user_name = args.user_name

    ip = args.flow_server_ip
    port = args.flow_server_port
    flow_client = FlowClient(ip=ip, port=port, version="v1")

    runtime_config = create_new_runtime_config()

    exec_toy_example(runtime_config)
