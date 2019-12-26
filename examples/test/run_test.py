import os
import json
from examples.test_example import submit
import argparse


def search_task(file_dir,suffix="testsuite.json"):
    task = []
    for root, dirs, files in os.walk(file_dir):
        for li in files:
            if li.endswith(suffix):
                path = os.path.join(root, li)
                print(path)
                task.append(path)
    return task


def role_map(env, role):
    _role, num = role.split("_", 1)
    loc = env["role"][_role][int(num)]
    return env["ip_map"][str(loc)]


def submit_task(env=None, task_data=None, task_conf=None, task_dsl=None, model=None, type=None):
    if task_data:
        for data in task_data:
            host = role_map(env, data["role"])
            if host != -1:
                submitter.run_upload(data_path=data["file"], config=data, remote_host=host)
            else:
                submitter.run_upload(data_path=data["file"], config=data)
    else:
        if type == "train":
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
    arg_parser.add_argument("env_file", type=str, help="please input your env config file")
    arg_parser.add_argument("result_file", type=str, help="please input the filename to receive results")
    arg_parser.add_argument("--name", type=str, help="please input the task name")
    args = arg_parser.parse_args()
    env_file = args.env_file
    result_file = args.result_file
    result = {}
    model = {}
    fate_home = os.path.abspath(f"{os.getcwd()}/../")
    submitter = submit.Submitter() \
        .set_fate_home(fate_home) \
        .set_work_mode(0)

    if env_file:
        with open(env_file) as e:
            env = json.loads(e.read())
            
    if args.name:
        task = search_task(os.path.join(fate_home, "federatedml-1.x-examples"),suffix=args.name)
    else:
        task = search_task(os.path.join(fate_home, "federatedml-1.x-examples"))

    for task_file in task:
        if task_file:
            # print(task_file)
            with open(task_file) as f:
                configs = json.loads(f.read())
        upload = submit_task(env=env, task_data=configs["data"])

        for task_name, config in configs["tasks"].items():
            try:
                conf = config["conf"]
                type = config["type"]
                if type == "train":
                    dsl = config["dsl"]
                    temp = submit_task(env=env, task_conf=conf, task_dsl=dsl, type=type)
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
