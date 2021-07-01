import json
import os
import argparse
import copy
import shutil
import subprocess

cur_dir = os.path.abspath(os.path.dirname(__file__))
print(f"cur_dir: {cur_dir}")


def insert_extract_code(file_path):
    code_lines = []
    code = \
    """
import json
import os
def extract(my_pipeline, file_name, output_path='generated_conf_and_dsl'):
    out_name = file_name.split('/')[-1]
    out_name = out_name.replace('pipeline-', '').replace('.py', '').replace('-', '_')
    conf = my_pipeline.get_train_conf()
    dsl = my_pipeline.get_train_dsl()
    cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    conf_name = os.path.join(cur_dir, output_path, f"{out_name}_conf.json")
    dsl_name = os.path.join(cur_dir, output_path, f"{out_name}_dsl.json")

    json.dump(conf, open(conf_name, 'w'), indent=4)
    print('conf name is {}'.format(conf_name))
    json.dump(dsl, open(dsl_name, 'w'), indent=4)
    print('dsl name is {}'.format(dsl_name))
    """
    code_lines.append(code)

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if ".predict(" in l:
                code_lines.append(f"# {l}")
            elif 'if __name__ == "__main__":' in l:
                code_lines.append("    extract(pipeline, __file__)\n")
                code_lines.append(l)
            elif "get_summary()" in l:
                continue
            else:
                code_lines.append(l)

    return code_lines


def extract(my_pipeline, file_name, output_path='generated_conf_and_dsl'):
    out_name = file_name.split('/')[-1]
    out_name = out_name.replace('pipeline-', '').replace('.py', '').replace('-', '_')
    conf = my_pipeline.get_train_conf()
    dsl = my_pipeline.get_train_dsl()
    conf_name = './{}/{}_conf.json'.format(output_path, out_name)
    dsl_name = './{}/{}_dsl.json'.format(output_path, out_name)
    json.dump(conf, open(conf_name, 'w'), indent=4)
    print('conf name is {}'.format(conf_name))
    json.dump(dsl, open(dsl_name, 'w'), indent=4)
    print('dsl name is {}'.format(dsl_name))


def get_testsuite_file(testsuite_file_path):
    # import examples
    # cpn_path = os.path.dirname(examples.__file__) + f'/federatedml-1.x-examples/{testsuite_file_path}'
    print(f"testsuite_file_path: {testsuite_file_path}")
    with open(testsuite_file_path, 'r', encoding='utf-8') as load_f:
        testsuite_json = json.load(load_f)
    # testsuite_json['tasks'] = {}
    if "tasks" in testsuite_json:
        del testsuite_json["tasks"]

    if "pipeline_tasks" in testsuite_json:
        del testsuite_json["pipeline_tasks"]
    return testsuite_json


def start_task(cmd):
    print('Start task: {}'.format(cmd))
    subp = subprocess.Popen(cmd,
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    # time_print("start_task, stdout:" + str(stdout))
    # try:
    #     stdout = json.loads(stdout)
    # except:
    #     raise RuntimeError("start task error, return value: {}".format(stdout))
    return stdout


def upload_data(file_path):
    cmd = ["fate_test", "suite", "-i", file_path, "--data-only", "--yes"]
    start_task(cmd)


def do_generated(file_path, fold_name, template_path, yaml_file):
    if not os.path.isdir(file_path):
        return
    files = os.listdir(file_path)
    # cmd = 'python {}'
    if template_path is None:
        for f in files:
            if "testsuite" in f and "generated_testsuite" not in f:
                template_path = os.path.join(file_path, f)
                break
    if template_path is None:
        # raise RuntimeError("Template cannot be found")
        return
    print(f"template_path: {template_path}")
    upload_data(file_path)
    suite_json = get_testsuite_file(template_path)
    pipeline_suite = copy.deepcopy(suite_json)
    suite_json["tasks"] = {}
    pipeline_suite["pipeline_tasks"] = {}
    replaced_path = os.path.join(file_path, 'replaced_code')
    generated_path = os.path.join(file_path, 'generated_conf_and_dsl')

    if not os.path.exists(replaced_path):
        os.system('mkdir {}'.format(replaced_path))

    if not os.path.exists(generated_path):
        os.system('mkdir {}'.format(generated_path))

    for f in files:
        if not f.startswith("pipeline"):
            continue
        # print(f)
        task_name = f.replace(".py", "")
        task_name = "-".join(task_name.split('-')[1:])
        pipeline_suite["pipeline_tasks"][task_name] = {
            "script": f
        }
        f_path = os.path.join(file_path, f)
        code_str = insert_extract_code(f_path)
        pipeline_file_path = os.path.join(replaced_path, f)
        open(pipeline_file_path, 'w').writelines(code_str)
        # print('replace done')
        # file_path = folder + f
        # os.system(cmd.format(folder + f))

    exe_files = os.listdir(replaced_path)
    #
    for f in exe_files:
        abs_file = os.path.join(replaced_path, f)
        print('executing {}'.format(abs_file))
        os.system(f"python {abs_file} -config {yaml_file}")

    conf_files = os.listdir(generated_path)
    f_dsl = {"_".join(f.split('_')[:-1]): f for f in conf_files if 'dsl.json' in f}
    f_conf = {"_".join(f.split('_')[:-1]): f for f in conf_files if 'conf.json' in f}

    for task_type, dsl_file in f_dsl.items():
        conf_file = f_conf[task_type]
        suite_json['tasks'][task_type] = {
            "conf": conf_file,
            "dsl": dsl_file
        }

    suite_path = os.path.join(generated_path, f"{fold_name}_testsuite.json")
    with open(suite_path, 'w', encoding='utf-8') as json_file:
        json.dump(suite_json, json_file, ensure_ascii=False, indent=4)

    suite_path = os.path.join(file_path, f"{fold_name}_pipeline_testsuite.json")
    with open(suite_path, 'w', encoding='utf-8') as json_file:
        json.dump(pipeline_suite, json_file, ensure_ascii=False, indent=4)

    shutil.rmtree(replaced_path)
    print("Generate testsuite and dsl&conf finished!")
    # os.system('rm -rf {}'.format(replaced_path))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--include", type=str, help="path of pipeline files", required=True)
    arg_parser.add_argument("-t", "--template_path", type=str, help="test template", required=False,
                            default=None)
    args = arg_parser.parse_args()

    input_path = args.include
    template_path = args.template_path
    input_path = os.path.abspath(input_path)
    input_list = [input_path]
    i = 0
    while i < len(input_list):
        dirs = os.listdir(input_list[i])
        for d in dirs:
            if os.path.isdir(d):
                input_list.append(d)
        i += 1

    yaml_file = os.path.join(os.path.dirname(cur_dir), "config.yaml")
    # print(file_path, module_name, template_path)
    for file_path in input_list:
        module_name = os.path.basename(file_path)
        do_generated(file_path, module_name, template_path, yaml_file)
    # pass
