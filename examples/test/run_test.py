import argparse
import json
import os
import time
import traceback

from examples.test import submit

TEST_SUITE_SUFFIX = "testsuite.json"


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


def data_upload(submitter, env, task_data, check_interval=3):
    for data in task_data:
        host = role_map(env, data["role"])
        remote_host = None if host == -1 else host
        format_msg = f"@{data['role']}:{data['file']} >> {data['namespace']}.{data['table_name']}"
        print(f"[{time.strftime('%Y-%m-%d %X')}]uploading {format_msg}")
        job_id = submitter.run_upload(data_path=data["file"], config=data, remote_host=remote_host)
        if not remote_host:
            submitter.await_finish(job_id, check_interval=check_interval)
        else:
            print("warning: not check remote uploading status!!!")
        print(f"[{time.strftime('%Y-%m-%d %X')}]upload done {format_msg}, job_id={job_id}\n")


def train_task(submitter, task_conf, task_dsl, task_name, check_interval=3):
    print(f"[{time.strftime('%Y-%m-%d %X')}][{task_name}]submitting...")
    output = submitter.submit_job(conf_temperate_path=task_conf, dsl_path=task_dsl)
    job_id = output['jobId']
    model_info = output['model_info']
    print(f"[{time.strftime('%Y-%m-%d %X')}][{task_name}]submit done, job_id={job_id}")
    status = submitter.await_finish(job_id, check_interval=check_interval, task_name=task_name)
    return dict(job_id=job_id, status=status, model_info=model_info)


def predict_task(submitter, task_conf, model, task_name, check_interval=3):
    print(f"[{time.strftime('%Y-%m-%d %X')}][{task_name}]submitting...")
    job_id = submitter.submit_pre_job(conf_temperate_path=task_conf, model_info=model)
    print(f"[{time.strftime('%Y-%m-%d %X')}][{task_name}]submit done, job_id={job_id}")
    ret = submitter.await_finish(job_id, check_interval=check_interval, task_name=task_name)
    return dict(job_id=job_id, status=ret)


def run_testsuite(submitter, env, file_name, err_name, check_interval=3, skip_data=False):
    result = {}
    model = {}
    testsuite_base_path = os.path.dirname(file_name)

    def check(field_name, config):
        if field_name not in config:
            raise ValueError(f"{field_name} not specified in {task_name}@{file_name}")

    with open(file_name) as f:
        configs = json.loads(f.read())

    if not skip_data:
        data_upload(submitter=submitter, env=env, task_data=configs["data"], check_interval=check_interval)

    for task_name, task_config in configs["tasks"].items():
        # noinspection PyBroadException
        try:
            check("conf", task_config)
            conf = os.path.join(testsuite_base_path, task_config["conf"])
            dep_task = task_config.get("deps", None)
            if dep_task is None:
                check("dsl", task_config)
                dsl = os.path.join(testsuite_base_path, task_config["dsl"])
                temp = train_task(submitter=submitter, task_conf=conf, task_dsl=dsl, task_name=task_name,
                                  check_interval=check_interval)
                job_id = temp['job_id']
                status = temp['status']
                model_info = temp["model_info"]
                result[task_name] = f"{job_id}\t{status}"
                model[task_name] = model_info
            else:
                temp = predict_task(submitter=submitter, task_conf=conf, model=model[dep_task], task_name=task_name,
                                    check_interval=check_interval)
                job_id = temp['job_id']
                status = temp['status']
                result[task_name] = f"{job_id}\t{status}"

        except Exception:
            print(f"[{time.strftime('%Y-%m-%d %X')}][{task_name}]task fail")
            err_msg = traceback.format_exc()
            print(err_msg)
            result[task_name] = "\tsubmit_fail"
            with open(f"{err_name}", "a") as f:
                f.write(f"{task_name}\n")
                f.write("===========\n")
                f.write(err_msg)
                f.write("\n")
    return result


def main():
    fate_home = os.path.abspath(f"{os.getcwd()}/../")
    example_path = "federatedml-1.x-examples"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("env_conf", type=str, help="file to read env config")
    arg_parser.add_argument("-o", "--output", type=str, help="file to save result, defaults to `test_result`",
                            default="test_result")
    arg_parser.add_argument("-e", "--error", type=str, help="file to save error")
    group = arg_parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--dir", type=str, help="dir to find testsuites",
                       default=os.path.join(fate_home, example_path))
    group.add_argument("-s", "--suite", type=str, help="testsuite to run")
    arg_parser.add_argument("-i", "--interval", type=int, help="check job status every i seconds, defaults to 1",
                            default=1)
    arg_parser.add_argument("--skip_data", help="skip data upload", action="store_true")
    args = arg_parser.parse_args()

    env_conf = args.env_conf
    output_file = args.output
    err_output = args.error or f"{output_file}.err"
    testsuites_dir = args.dir
    suite = args.suite
    interval = args.interval
    skip_data = args.skip_data

    submitter = submit.Submitter(fate_home=fate_home, work_mode=0)

    try:
        with open(env_conf) as e:
            env = json.loads(e.read())
    except:
        raise ValueError(f"invalid env conf: {env_conf}")
    testsuites = [suite] if suite else search_testsuite(testsuites_dir)

    for file_name in testsuites:
        print("===========================================")
        print(f"[{time.strftime('%Y-%m-%d %X')}]running testsuite {file_name}")
        print("===========================================")
        result = run_testsuite(submitter, env, file_name, err_output,
                               check_interval=interval,
                               skip_data=skip_data)

        with open(output_file, "w") as f:
            f.write("===========================================\n")
            f.write(f"{file_name}\n")
            f.write("===========================================\n")
            for task_name, task_status in result.items():
                f.write(f"{task_name}\t{task_status}\n")


if __name__ == "__main__":
    main()
