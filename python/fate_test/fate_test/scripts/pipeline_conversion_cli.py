import copy
import os
import shutil
import sys
import time
import uuid
import json
import click
import importlib

from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions


@click.group(name="convert")
def convert_group():
    """
    Converting pipeline files to dsl v2
    """
    ...


@convert_group.command("pipeline-to-dsl")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *pipeline.py under these paths")
@click.option('-o', '--output-path', type=click.Path(exists=True), help="DSL output path, default to *pipeline.py path")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def to_dsl(ctx, include, output_path, **kwargs):
    """
    This command will run pipeline, make sure data is uploaded
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"converting namespace: {namespace}", fg='red')
    for path in include:
        echo.echo(f"pipeline path: {os.path.abspath(path)}")
    if not yes and not click.confirm("running?"):
        return
    config_yaml_file = './examples/config.yaml'
    temp_file_path = f'./logs/{namespace}/temp_pipeline.py'

    for i in include:
        try:
            convert(i, temp_file_path, config_yaml_file, output_path, config_inst)
        except Exception:
            exception_id = uuid.uuid1()
            echo.echo(f"exception_id={exception_id}")
            LOGGER.exception(f"exception id: {exception_id}")
        finally:
            echo.stdout_newline()
    echo.farewell()
    echo.echo(f"converting namespace: {namespace}", fg='red')


@convert_group.command("pipeline-testsuite-to-dsl-testsuite")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), metavar="<include>",
              help="include is the pipeline test folder containing *testsuite.py")
@click.option('-t', '--template-path', required=False, type=click.Path(exists=True), metavar="<include>",
              help="specify the test template to use")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def to_testsuite(ctx, include, template_path, **kwargs):
    """
    convert pipeline testsuite to dsl testsuite
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]
    echo.welcome()
    if not os.path.isdir(include):
        raise Exception("Please fill in a folder.")
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo(f"pipeline path: {os.path.abspath(include)}")
    if not yes and not click.confirm("running?"):
        return
    input_path = os.path.abspath(include)
    input_list = [input_path]
    i = 0
    while i < len(input_list):
        dirs = os.listdir(input_list[i])
        for d in dirs:
            if os.path.isdir(d):
                input_list.append(d)
        i += 1

    for file_path in input_list:
        try:
            module_name = os.path.basename(file_path)
            do_generated(file_path, module_name, template_path, config_inst)
        except Exception:
            exception_id = uuid.uuid1()
            echo.echo(f"exception_id={exception_id}")
            LOGGER.exception(f"exception id: {exception_id}")
        finally:
            echo.stdout_newline()
    echo.farewell()
    echo.echo(f"converting namespace: {namespace}", fg='red')


def make_temp_pipeline(pipeline_file, temp_file_path, folder_name):
    def _conf_file_update(_line, k, end, conf_file=None):
        if ")" in _line[0]:
            if conf_file is None:
                conf_file = os.path.abspath(folder_name + "/" + _line[0].replace("'", "").replace('"', "").
                                            replace(")", "").replace(":", "").replace("\n", ""))
            _line = k + conf_file + end
        else:
            if conf_file is None:
                conf_file = os.path.abspath(folder_name + "/" + _line[0].replace('"', ""))
            _line = k + conf_file + '",' + _line[-1]

        return conf_file, _line

    def _get_conf_file(_lines):
        param_default = False
        conf_file = None
        for _line in _lines:
            if "--param" in _line or param_default:
                if "default" in _line:
                    _line_start = _line.split("default=")
                    _line_end = _line_start[1].split(",")
                    conf_file, _ = _conf_file_update(_line_end, 'default="', '")')
                    param_default = False
                else:
                    param_default = True
        return conf_file

    code_list = []
    with open(pipeline_file, 'r') as f:
        lines = f.readlines()
        start_main = False
        has_returned = False
        space_num = 0
        conf_file_dir = _get_conf_file(lines)
        for line in lines:
            if line is None:
                continue
            elif "def main" in line:
                for char in line:
                    if char.isspace():
                        space_num += 1
                    else:
                        break
                start_main = True
                if "param=" in line:
                    line_start = line.split("param=")
                    line_end = line_start[1].split(",")
                    conf_file_dir, line = _conf_file_update(line_end, 'param="', '")', conf_file_dir)
                    line = line_start[0] + line
            elif start_main and "def " in line and not has_returned:
                code_list.append(" " * (space_num + 4) + "return pipeline\n")
                start_main = False
            elif start_main and "return " in line:
                code_list.append(" " * (space_num + 4) + "return pipeline\n")
                start_main = False
                continue
            elif start_main and 'if __name__ ==' in line:
                code_list.append(" " * (space_num + 4) + "return pipeline\n")
                start_main = False
            code_list.append(line)
        if start_main:
            code_list.append(" " * (space_num + 4) + "return pipeline\n")

    with open(temp_file_path, 'w') as f:
        f.writelines(code_list)


def convert(pipeline_file, temp_file_path, config_yaml_file, output_path, config: Config):
    folder_name, file_name = os.path.split(pipeline_file)
    if output_path is not None:
        folder_name = output_path
    echo.echo(f"folder_name: {os.path.abspath(folder_name)}, file_name: {file_name}")
    conf_name = file_name.replace('.py', '_conf.json')
    dsl_name = file_name.replace('.py', '_dsl.json')
    conf_name = os.path.join(folder_name, conf_name)
    dsl_name = os.path.join(folder_name, dsl_name)

    make_temp_pipeline(pipeline_file, temp_file_path, folder_name)
    additional_path = os.path.realpath(os.path.join(os.path.curdir, pipeline_file, os.pardir, os.pardir))
    if additional_path not in sys.path:
        sys.path.append(additional_path)
    loader = importlib.machinery.SourceFileLoader("main", str(temp_file_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    my_pipeline = mod.main(os.path.join(config.data_base_dir, config_yaml_file))
    conf = my_pipeline.get_train_conf()
    dsl = my_pipeline.get_train_dsl()
    os.remove(temp_file_path)

    with open(conf_name, 'w') as f:
        json.dump(conf, f, indent=4)
        echo.echo('conf name is {}'.format(os.path.abspath(conf_name)))
    with open(dsl_name, 'w') as f:
        json.dump(dsl, f, indent=4)
        echo.echo('dsl name is {}'.format(os.path.abspath(dsl_name)))


def insert_extract_code(file_path):
    code_lines = []
    code = \
        """
import json
import os
def extract(my_pipeline, file_name, output_path='dsl_testsuite'):
    out_name = file_name.split('/')[-1]
    out_name = out_name.replace('pipeline-', '').replace('.py', '').replace('-', '_')
    conf = my_pipeline.get_train_conf()
    dsl = my_pipeline.get_train_dsl()
    cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    conf_name = os.path.join(cur_dir, output_path, f"{out_name}_conf.json")
    dsl_name = os.path.join(cur_dir, output_path, f"{out_name}_dsl.json")
    json.dump(conf, open(conf_name, 'w'), indent=4)
    json.dump(dsl, open(dsl_name, 'w'), indent=4)
        """

    code_lines.append(code)
    screen_keywords = [".predict(", ".fit(", ".deploy_component(", "predict_pipeline ",
                       "predict_pipeline."]
    continue_to_screen = False
    has_return = False

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if ".predict(" in l or ".fit(" in l:
                code_lines.append(f"# {l}")

            elif 'if __name__ == "__main__":' in l:
                if not has_return:
                    code_lines.append("    extract(pipeline, __file__)\n")
                code_lines.append(l)

            elif 'return' in l:
                code_lines.append("    extract(pipeline, __file__)\n")
                # code_lines.append(l)
                has_return = True

            elif "get_summary()" in l:
                continue
            elif continue_to_screen:
                code_lines.append(f"# {l}")
                if ")" in l:
                    continue_to_screen = False
            else:
                should_append = True
                for key_word in screen_keywords:
                    if key_word in l:
                        code_lines.append(f"# {l}")
                        should_append = False
                        if ")" not in l:
                            continue_to_screen = True
                if should_append:
                    code_lines.append(l)

    return code_lines


def get_testsuite_file(testsuite_file_path):
    echo.echo(f"testsuite_file_path: {testsuite_file_path}")
    with open(testsuite_file_path, 'r', encoding='utf-8') as load_f:
        testsuite_json = json.load(load_f)
    if "tasks" in testsuite_json:
        del testsuite_json["tasks"]
    if "pipeline_tasks" in testsuite_json:
        del testsuite_json["pipeline_tasks"]
    return testsuite_json


def do_generated(file_path, fold_name, template_path, config: Config):
    yaml_file = os.path.join(config.data_base_dir, "./examples/config.yaml")
    PYTHONPATH = os.environ.get('PYTHONPATH') + ":" + str(config.data_base_dir)
    os.environ['PYTHONPATH'] = PYTHONPATH
    if not os.path.isdir(file_path):
        return
    files = os.listdir(file_path)
    if template_path is None:
        for f in files:
            if "testsuite" in f and "generated_testsuite" not in f:
                template_path = os.path.join(file_path, f)
                break
    if template_path is None:
        return

    suite_json = get_testsuite_file(template_path)
    pipeline_suite = copy.deepcopy(suite_json)
    suite_json["tasks"] = {}
    pipeline_suite["pipeline_tasks"] = {}
    replaced_path = os.path.join(file_path, 'replaced_code')
    generated_path = os.path.join(file_path, 'dsl_testsuite')

    if not os.path.exists(replaced_path):
        os.system('mkdir {}'.format(replaced_path))

    if not os.path.exists(generated_path):
        os.system('mkdir {}'.format(generated_path))

    for f in files:
        if not f.startswith("pipeline"):
            continue
        echo.echo(f)
        task_name = f.replace(".py", "")
        task_name = "-".join(task_name.split('-')[1:])
        pipeline_suite["pipeline_tasks"][task_name] = {
            "script": f
        }
        f_path = os.path.join(file_path, f)
        code_str = insert_extract_code(f_path)
        pipeline_file_path = os.path.join(replaced_path, f)
        open(pipeline_file_path, 'w').writelines(code_str)

    exe_files = os.listdir(replaced_path)
    fail_job_count = 0
    task_type_list = []
    exe_conf_file = None
    exe_dsl_file = None
    for i, f in enumerate(exe_files):
        abs_file = os.path.join(replaced_path, f)
        echo.echo('\n' + '[{}/{}]  executing {}'.format(i + 1, len(exe_files), abs_file), fg='red')
        result = os.system(f"python {abs_file} -config {yaml_file}")
        if not result:
            time.sleep(3)
            conf_files = os.listdir(generated_path)
            f_dsl = {"_".join(f.split('_')[:-1]): f for f in conf_files if 'dsl.json' in f}
            f_conf = {"_".join(f.split('_')[:-1]): f for f in conf_files if 'conf.json' in f}
            for task_type, dsl_file in f_dsl.items():
                if task_type not in task_type_list:
                    exe_dsl_file = dsl_file
                    task_type_list.append(task_type)
                    exe_conf_file = f_conf[task_type]
                    suite_json['tasks'][task_type] = {
                        "conf": exe_conf_file,
                        "dsl": exe_dsl_file
                    }
            echo.echo('conf name is {}'.format(os.path.join(file_path, "dsl_testsuite", exe_conf_file)))
            echo.echo('dsl name is {}'.format(os.path.join(file_path, "dsl_testsuite", exe_dsl_file)))
        else:
            echo.echo('profile generation failed')
            fail_job_count += 1

    suite_path = os.path.join(generated_path, f"{fold_name}_testsuite.json")
    with open(suite_path, 'w', encoding='utf-8') as json_file:
        json.dump(suite_json, json_file, ensure_ascii=False, indent=4)

    suite_path = os.path.join(file_path, f"{fold_name}_pipeline_testsuite.json")
    with open(suite_path, 'w', encoding='utf-8') as json_file:
        json.dump(pipeline_suite, json_file, ensure_ascii=False, indent=4)

    shutil.rmtree(replaced_path)
    if not fail_job_count:
        echo.echo("Generate testsuite and dsl&conf finished!")
    else:
        echo.echo("Generate testsuite and dsl&conf finished! {} failures".format(fail_job_count))
