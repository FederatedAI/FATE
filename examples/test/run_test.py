import argparse
import json
import os

from examples.test import submit

TEST_SUITE_SUFFIX = "testsuite.json"
EXAMPLE_PATH = "federatedml-1.x-examples"


def search_testsuite(file_dir, suffix=TEST_SUITE_SUFFIX):
    testsuites = []
    for root, dirs, files in os.walk(file_dir):
        for file_name in files:
            if file_name.endswith(suffix):
                path = os.path.join(root, file_name)
                testsuites.append(path)
    return testsuites


def role_map(env, role):
    _role, num = role.split("_", 1)
    loc = env["role"][_role][int(num)]
    return env["ip_map"][str(loc)]


def data_upload(submitter, env, task_data):
    for data in task_data:
        host = role_map(env, data["role"])
        remote_host = None if host == -1 else host
        submitter.run_upload(data_path=data["file"], config=data, remote_host=remote_host)


def train_task(submitter, task_conf, task_dsl):
    output = submitter.submit_job(conf_temperate_path=task_conf, dsl_path=task_dsl)
    ret = submitter.await_finish(output["jobId"])
    return dict(output=output, status=ret)


def predict_task(submitter, task_conf, model=None):
    job_id = submitter.submit_pre_job(conf_temperate_path=task_conf, model_info=model)
    ret = submitter.await_finish(job_id)
    return dict(job_id=job_id, status=ret)


def run_testsuite(submitter, env, file_name, err_name):
    result = {}
    model = {}
    testsuite_base_path = os.path.dirname(file_name)

    def check(field_name, config):
        if field_name not in config:
            raise ValueError(f"{field_name} not specified in {task_name}@{file_name}")

    with open(file_name) as f:
        configs = json.loads(f.read())

    data_upload(submitter=submitter, env=env, task_data=configs["data"])

    for task_name, task_config in configs["tasks"].items():
        try:
            check("conf", task_config)
            conf = os.path.join(testsuite_base_path, task_config["conf"])
            task_type = task_config.get("type", "train")

            if task_type == "train":
                check("dsl", task_config)
                dsl = os.path.join(testsuite_base_path, task_config["dsl"])
                temp = train_task(submitter=submitter, task_conf=conf, task_dsl=dsl)
                result[task_name] = f"{temp['output']['jobId']}\t{temp['status']}"
                model[task_name] = temp["output"]["model_info"]
            elif task_type == "predict":
                check("task", task_config)
                pre_task = task_config["task"]
                temp = predict_task(submitter=submitter, task_conf=conf, model=model[pre_task])
                result[task_name] = f"{temp['job_id']}\t{temp['status']}"
        except Exception as e:
            result[task_name] = "\tsubmit_fail"
            with open(f"{err_name}.err", "a") as f:
                f.write(f"{task_name}\n")
                f.write("===========\n")
                f.write(json.dumps(e.args))
                f.write("\n")
    return result


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("env_file", type=str, help="please input your env config file")
    arg_parser.add_argument("result_file", type=str, help="please input the filename to receive results")
    arg_parser.add_argument("--name", type=str, help="please input the task name")
    args = arg_parser.parse_args()

    env_file = args.env_file
    result_file = args.result_file
    fate_home = os.path.abspath(f"{os.getcwd()}/../")
    submitter = submit.Submitter(fate_home=fate_home, work_mode=0)

    try:
        with open(env_file) as e:
            env = json.loads(e.read())
    except:
        raise ValueError(f"invalid env file: {env_file}")

    if args.name is not None:
        testsuites = [args.name]
    else:
        testsuites = search_testsuite(os.path.join(fate_home, EXAMPLE_PATH))

    for file_name in testsuites:
        result = run_testsuite(submitter, env, file_name, result_file)

        with open(result_file, "w") as f:
            f.write("===========================================\n")
            f.write(f"{file_name}\n")
            f.write("===========================================\n")
            for task_name, task_status in result.items():
                f.write(f"{task_name}\t{task_status}\n")


if __name__ == "__main__":
    main()
