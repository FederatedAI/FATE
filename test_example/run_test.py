import os
import json
from test_example import submit
import argparse


def submit_task(task_data, task_conf, task_dsl):
    for data in task_data:
        if "@" in data:
            path, host = data.split("@", 1)
            submitter.run_upload(upload_path=path, remote_host=host)
        else:
            submitter.run_upload(upload_path=data)
    job_id = submitter.submit_job(conf_temperate_path=task_conf, dsl_path=task_dsl)
    ret = submitter.await_finish(job_id)
    return job_id+ret


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("task_file", type=str, help="please input your task config file")
    arg_parser.add_argument("result_file", type=str, help="please input the filename to receive results")
    args = arg_parser.parse_args()
    task_file=args.task_file
    result_file=args.result_file
    if task_file:
       with open(task_file) as f:
            configs = json.loads(f.read())

    result = {}
    fate_home = os.path.abspath(f"{os.getcwd()}/../")
    submitter = submit.Submitter() \
        .set_fate_home(fate_home) \
        .set_work_mode(0)

    for task_name, config in configs.items():
        try:
            upload_data = config["data"]
            conf = config["conf"]
            dsl = config["dsl"]
            result[task_name] = submit_task(task_data=upload_data, task_conf=conf, task_dsl=dsl)
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
