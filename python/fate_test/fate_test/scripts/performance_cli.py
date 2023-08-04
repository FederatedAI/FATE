#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import glob
import os
import time
import uuid
from datetime import timedelta
from inspect import signature
from ruamel import yaml

import click
from fate_test._client import Clients
from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test._parser import PerformanceSuite
from fate_test.scripts._options import SharedOptions
from fate_test.scripts._utils import _load_testsuites, _upload_data, _delete_data, _load_module_from_script, \
    _add_replace_hook
from fate_test.utils import TxtStyle, parse_job_time_info, pretty_time_info_summary
from prettytable import PrettyTable, ORGMODE


@click.command("performance")
@click.option('-t', '--job-type', type=click.Choice(['intersect', 'intersect_multi', 'hetero_lr', 'hetero_sbt']),
              help="Select the job type, you can also set through include")
@click.option('-i', '--include', type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *performance.yaml under these paths")
@click.option('-m', '--timeout', type=int,
              help="maximum running time of job")
@click.option('-e', '--epochs', type=int, help="When the algorithm model is LR, the number of iterations is set")
@click.option('-d', '--max-depth', type=int,
              help="When the algorithm model is SecureBoost, set the number of model layers")
@click.option('-nt', '--num-trees', type=int, help="When the algorithm model is SecureBoost, set the number of trees")
@click.option('-p', '--task-cores', type=int, help="processors per node")
@click.option('-s', '--storage-tag', type=str,
              help="tag for storing performance time consuming, for future comparison")
@click.option('-v', '--history-tag', type=str, multiple=True,
              help="Extract performance time consuming from history tags for comparison")
@click.option("--skip-data", is_flag=True, default=False,
              help="skip uploading data specified in testsuite")
@click.option("--provider", type=str,
              help="Select the fate version, for example: fate@1.7")
@click.option("--disable-clean-data", "clean_data", flag_value=False, default=None)
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def run_task(ctx, job_type, include, replace, timeout, epochs,
             max_depth, num_trees, task_cores, storage_tag, history_tag, skip_data, clean_data, provider, **kwargs):
    """
    Test the performance of big data tasks, alias: bp
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    config_inst = ctx.obj["config"]
    if ctx.obj["extend_sid"] is not None:
        config_inst.extend_sid = ctx.obj["extend_sid"]
    if task_cores is not None:
        config_inst.update_conf(task_cores=task_cores)
    if timeout is not None:
        config_inst.update_conf(timeout=timeout)
    """if ctx.obj["auto_increasing_sid"] is not None:
        config_inst.auto_increasing_sid = ctx.obj["auto_increasing_sid"]"""
    namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]
    data_namespace_mangling = ctx.obj["namespace_mangling"]
    if clean_data is None:
        clean_data = config_inst.clean_data

    def get_perf_template(conf: Config, job_type):
        perf_dir = os.path.join(os.path.abspath(conf.perf_template_dir) + '/' + job_type + '/' + "*testsuite.yaml")
        return glob.glob(perf_dir)

    if not include:
        include = get_perf_template(config_inst, job_type)
    # prepare output dir and json hooks
    _add_replace_hook(replace)

    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    suites = _load_testsuites(includes=include, excludes=tuple(), glob=None, provider=provider,
                              suffix="performance.yaml", suite_type="performance")
    for i, suite in enumerate(suites):
        echo.echo(f"\tdataset({len(suite.dataset)}) dsl jobs({len(suite.jobs)}) {suite.path}")

    if not yes and not click.confirm("running?"):
        return

    echo.stdout_newline()
    client = Clients(config_inst)

    for i, suite in enumerate(suites):
        # noinspection PyBroadException
        try:
            start = time.time()
            echo.echo(f"[{i + 1}/{len(suites)}]start at {time.strftime('%Y-%m-%d %X')} {suite.path}", fg='red')

            if not skip_data:
                try:
                    _upload_data(client, suite, config_inst)
                except Exception as e:
                    raise RuntimeError(f"exception occur while uploading data for {suite.path}") from e

            echo.stdout_newline()
            try:
                job_time_info = _run_performance_jobs(config_inst, suite, namespace, data_namespace_mangling, client,
                                                      epochs, max_depth, num_trees)
            except Exception as e:
                raise RuntimeError(f"exception occur while running pipeline jobs for {suite.path}") from e

            echo.echo(f"[{i + 1}/{len(suites)}]elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
            if not skip_data and clean_data:
                _delete_data(client, suite)
            # echo.echo(suite.pretty_final_summary(job_time_info), fg='red')
            all_summary = []
            compare_summary = []
            for job_name, job_time in job_time_info.items():
                performance_dir = "/".join(
                    [os.path.join(os.path.abspath(config_inst.cache_directory),
                                  'benchmark_history', "performance.yaml")])
                # @todo: change to client query result
                # fate_version = clients["guest_0"].get_version()
                fate_version = "beta-2.0.0"
                if history_tag:
                    history_tag = ["_".join([i, job_name]) for i in history_tag]
                    history_compare_result = comparison_quality(job_name,
                                                                history_tag,
                                                                performance_dir,
                                                                job_time["time_summary"])
                    compare_summary.append(history_compare_result)
                if storage_tag:
                    storage_tag = "_".join(['FATE', fate_version, storage_tag, job_name])
                    save_quality(storage_tag, performance_dir, job_time["time_summary"])
                res_str = pretty_time_info_summary(job_time, job_name)
                all_summary.append(res_str)
            echo.echo("\n".join(all_summary))
            echo.echo("#" * 60)
            echo.echo("\n".join(compare_summary))

            echo.echo()

        except Exception:
            exception_id = uuid.uuid1()
            echo.echo(f"exception in {suite.path}, exception_id={exception_id}")
            LOGGER.exception(f"exception id: {exception_id}")
        finally:
            echo.stdout_newline()

    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


@LOGGER.catch
def _run_performance_jobs(config: Config, suite: PerformanceSuite, tol: float, namespace: str,
                          data_namespace_mangling: bool, client, epochs, max_depth, num_trees):
    # pipeline demo goes here
    job_n = len(suite.pipeline_jobs)
    fate_base = config.fate_base
    PYTHONPATH = os.environ.get('PYTHONPATH') + ":" + os.path.join(fate_base, "python")
    os.environ['PYTHONPATH'] = PYTHONPATH
    job_time_history = {}
    for j, job in enumerate(suite.pipeline_jobs):
        try:
            echo.echo(f"Running [{j + 1}/{job_n}] job: {job.job_name}")
            job_name, script_path, conf_path = job.job_name, job.script_path, job.conf_path
            param = Config.load_from_file(conf_path)
            if epochs is not None:
                param['epochs'] = epochs
            if max_depth is not None:
                param['max_depth'] = max_depth
            if num_trees is not None:
                param['num_trees'] = num_trees

            mod = _load_module_from_script(script_path)
            input_params = signature(mod.main).parameters
            # local script
            if len(input_params) == 1:
                job_id = mod.main(param=param)
            elif len(input_params) == 2:
                job_id = mod.main(config=config, param=param)
            # pipeline script
            elif len(input_params) == 3:
                if data_namespace_mangling:
                    job_id = mod.main(config=config, param=param, namespace=f"_{namespace}")
                else:
                    job_id = mod.main(config=config, param=param)
            else:
                job_id = mod.main()
            echo.echo(f"[{j + 1}/{job_n}] job: {job.job_name} Success!\n")
            ret_msg = client.query_time_elapse(job_id, role="guest", party_id=config.parties.guest[0]).get("data")
            time_summary = parse_job_time_info(ret_msg)
            job_time_history[job_name] = {"job_id": job_id, "time_summary": time_summary}
            echo.echo(f"[{j + 1}/{job_n}] job: {job.job_name} time info: {time_summary}\n")

        except Exception as e:
            exception_id = uuid.uuid1()
            echo.echo(f"exception while running [{j + 1}/{job_n}] job, exception_id={exception_id}", err=True,
                      fg='red')
            LOGGER.exception(f"exception id: {exception_id}, error message: \n{e}")
            continue
    return job_time_history


def comparison_quality(group_name, history_tags, history_info_dir, time_consuming):
    assert os.path.exists(history_info_dir), f"Please check the {history_info_dir} Is it deleted"
    with open(history_info_dir, 'r') as f:
        benchmark_quality = yaml.load(f)
    benchmark_performance = {}
    table = PrettyTable()
    table.set_style(ORGMODE)
    table.field_names = ["Script Model Name", "component", "time consuming"]
    for history_tag in history_tags:
        for tag in benchmark_quality:
            if '_'.join(tag.split("_")[2:]) == history_tag:
                benchmark_performance[tag] = benchmark_quality[tag]
    if benchmark_performance is not None:
        benchmark_performance[group_name] = time_consuming

    for script_model_name in benchmark_performance:
        for cpn, time in benchmark_performance[script_model_name].items():
            table.add_row([f"{script_model_name}"] +
                          [f"{TxtStyle.FIELD_VAL}{cpn}{TxtStyle.END}"] +
                          [f"{TxtStyle.FIELD_VAL}{time}{TxtStyle.END}"])
    # print("\n")
    # print(table.get_string(title=f"{TxtStyle.TITLE}Performance comparison results{TxtStyle.END}"))
    # print("#" * 60)
    return table.get_string(title=f"{TxtStyle.TITLE}Performance comparison results{TxtStyle.END}")


def save_quality(storage_tag, save_dir, time_consuming):
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    if os.path.exists(save_dir):
        with open(save_dir, 'r') as f:
            benchmark_quality = yaml.load(f)
    else:
        benchmark_quality = {}
    benchmark_quality.update({storage_tag: time_consuming})
    try:
        with open(save_dir, 'w') as fp:
            yaml.dump(benchmark_quality, fp)
        print("\n" + "Storage successful, please check: ", save_dir)
    except Exception:
        print("\n" + "Storage failed, please check: ", save_dir)
