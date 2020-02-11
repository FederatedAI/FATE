import argparse
import json
import os
import random
import subprocess
import sys
import time

home_dir = os.path.split(os.path.realpath(__file__))[0]
fate_flow_path = os.path.join(home_dir, "..", "..", "fate_flow", "fate_flow_client.py")
dsl_path = os.path.join(home_dir, "toy_example_dsl.json")
conf_path = os.path.join(home_dir, "toy_example_conf.json")

guest_party_id = -1
host_party_id = -1

work_mode = 0

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
    with open(conf_path, "r") as fin:
        conf_dict = json.loads(fin.read())

    if not conf_dict:
        if not os.path.isfile(conf_dict):
            raise ValueError("config file {} dose not exist, please check!".format(conf_path))

        raise ValueError("{} ")

    conf_dict["initiator"]["party_id"] = guest_party_id
    conf_dict["job_parameters"]["work_mode"] = work_mode
    conf_dict["role"]["guest"] = [guest_party_id]
    conf_dict["role"]["host"] = [host_party_id]

    new_config_path = gen_unique_path()

    with open(new_config_path, "w") as fout:
        json_str = json.dumps(conf_dict, indent=1)
        fout.write(json_str + "\n")

    return new_config_path


def exec_task(dsl_path, config_path):
    subp = subprocess.Popen(["python",
                             fate_flow_path,
                             "-f",
                             "submit_job",
                             "-d",
                             dsl_path,
                             "-c",
                             config_path],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    print("stdout:" + str(stdout))

    status = -1
    try:
        stdout = json.loads(stdout)
        status = stdout["retcode"]
    except:
        raise ValueError("failed to exec task, stderr is {}, stdout is {}".format(stderr, stdout))

    if status != 0:
        raise ValueError(
            "failed to exec task, status:{}, stderr is {} stdout:{}".format(status, stderr, stdout))

    jobid = stdout["jobId"]
    return jobid


def get_job_status(jobid):
    subp = subprocess.Popen(["python",
                             fate_flow_path,
                             "-f",
                             "query_job",
                             "-j",
                             jobid,
                             "-r",
                             "guest"],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")

    retcode = -1
    try:
        stdout = json.loads(stdout)
        retcode = stdout["retcode"]
    except:
        return "query job status failed"

    if retcode != 0:
        return "query job status failed"

    status = stdout["data"][0]["f_status"]
    return status


def show_log(jobid, log_level):
    log_dir = os.path.join(home_dir, "test", "log")
    log_path = os.path.join(log_dir, "job_" + jobid + "_log", "guest", str(guest_party_id), "secure_add_example_0")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    subp = subprocess.Popen(["python",
                             fate_flow_path,
                             "-f",
                             "job_log",
                             "-j",
                             jobid,
                             "-o",
                             log_dir],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    subp.communicate()
    if log_level == "error":
        error_log = os.path.join(log_path, "ERROR.log")
        
        with open(error_log, "r") as fin:
            for line in fin:
                print (line.strip())
    else:
        info_log = os.path.join(log_path, "INFO.log")
        with open(info_log, "r") as fin:
            for line in fin:
                if line.find("secure_add_guest.py[line") != -1:
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
            print ("job status is {}".format((status)))

    raise ValueError("job running time exceed, please check federation or eggroll log")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("guest_party_id", type=int, help="please input guest party id")
    arg_parser.add_argument("host_party_id", type=int, help="please input host party id")
    arg_parser.add_argument("work_mode", type=int,
                            help="please input work_mode, 0 stands for standalone, 1  stands for cluster")

    args = arg_parser.parse_args()

    guest_party_id = args.guest_party_id
    host_party_id = args.host_party_id
    work_mode = args.work_mode

    runtime_config = create_new_runtime_config()

    exec_toy_example(runtime_config)
