import os
import json
from test_example import submit
import argparse


def role_map(env, role):
    _role, num = role.split("_", 1)
    loc = env["role"][_role][int(num)]
    return env["ip_map"][str(loc)]


def submit_task(env=None, task_data=None, task_conf=None, task_dsl=None, model=None, type=None):
    if type == "train":
        for data in task_data:
            host = role_map(env, data["role"])
            if host != -1:
                submitter.run_upload(data_path=data["file"], config=data, remote_host=host)
            else:
                submitter.run_upload(data_path=data["file"], config=data)
        output = submitter.submit_job(conf_temperate_path=task_conf, dsl_path=task_dsl)
        ret = submitter.await_finish(output["jobId"])
        res = {}
        res['output'] = output
        res['status'] = ret
        return res
    else:
        job_id = submitter.submit_pre_job(conf_temperate_path=task_conf, model_info=model)
        ret = submitter.await_finish(job_id)
        return job_id + ret


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("task_file", type=str, help="please input your task config file")
    arg_parser.add_argument("result_file", type=str, help="please input the filename to receive results")
    arg_parser.add_argument("work_mode", type=int, help="please input your work_mode")
    args = arg_parser.parse_args()
    task_file = args.task_file
    result_file = args.result_file
    work_mode = args.work_mode
    if task_file:
        with open(task_file) as f:
            configs = json.loads(f.read())

    result = {}
    model = {}
    fate_home = os.path.abspath(f"{os.getcwd()}/../")
    submitter = submit.Submitter() \
        .set_fate_home(fate_home) \
        .set_work_mode(work_mode)

    env = configs["env"]

    for task_name, config in configs["tasks"].items():
        try:
            conf = config["conf"]
            type = config["type"]
            if type == "train":
                upload_data = config["data"]
                dsl = config["dsl"]
                temp = submit_task(env=env, task_data=upload_data, task_conf=conf, task_dsl=dsl, type=type)
                result[task_name] = temp["output"]["jobId"] + temp["status"]
                model[task_name] = temp["output"]["model_info"]
            else:
                task = config["task"]
                result[task_name] = submit_task(task_conf=conf, model=model[task], type=type)
        except Exception as e:
            result[task_name] = "submit_fail"
            with open("err.txt", "a") as f:
                f.write(f"{task_name}\n")
                f.write("===========")
                f.write(json.dumps(e.args))
                f.write("\n")
    with open(result_file, "w") as f:
        f.write("task\tstatus\n")
        f.write("____________________________\n")
        for task_name, task_status in result.items():
            f.write(f"{task_name}\t{task_status}\n")
