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
import time
import uuid
from datetime import timedelta

import click
from fate_test._client import Clients
from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test._parser import Testsuite, non_success_summary
from fate_test.scripts._options import SharedOptions
from fate_test.scripts._utils import _load_testsuites, _upload_data, _delete_data, _load_module_from_script

from fate_test import _config

"""
@click.option('-uj', '--update-job-parameters', default="{}", type=JSON_STRING,
              help="a json string represents mapping for replacing fields in conf.job_parameters")
@click.option('-uc', '--update-component-parameters', default="{}", type=JSON_STRING,
              help="a json string represents mapping for replacing fields in conf.component_parameters")
@click.option('-m', '--timeout', type=int, default=3600, help="maximun running time of job")
@click.option('-p', '--task-cores', type=int, help="processors per node")
"""


@click.command("suite")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *testsuite.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *testsuite.json under these paths")
@click.option("-g", '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option("--skip-jobs", is_flag=True, default=False,
              help="skip pipeline jobs defined in testsuite")
@click.option("--skip-data", is_flag=True, default=False,
              help="skip uploading data specified in testsuite")
@click.option("--data-only", is_flag=True, default=False,
              help="upload data only")
@click.option("--provider", type=str,
              help="Select the fate version, for example: fate@2.0-beta")
@click.option("--disable-clean-data", "clean_data", flag_value=False, default=None)
@click.option("--enable-clean-data", "clean_data", flag_value=True, default=None)
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def run_suite(ctx, include, exclude, glob,
              skip_jobs, skip_data, data_only, clean_data, provider, **kwargs):
    """
    process testsuite
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    config_inst = ctx.obj["config"]
    """if ctx.obj["extend_sid"] is not None:
        config_inst.extend_sid = ctx.obj["extend_sid"]
    if ctx.obj["auto_increasing_sid"] is not None:
        config_inst.auto_increasing_sid = ctx.obj["auto_increasing_sid"]"""
    if clean_data is None:
        clean_data = config_inst.clean_data
    namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]
    data_namespace_mangling = ctx.obj["namespace_mangling"]
    # prepare output dir and json hooks
    # _add_replace_hook(replace)
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    suites = _load_testsuites(includes=include, excludes=exclude, glob=glob, provider=provider)
    for suite in suites:
        _config.jobs_num += len(suite.pipeline_jobs)
        echo.echo(f"\tdataset({len(suite.dataset)}) "
                  f"pipeline jobs ({len(suite.pipeline_jobs)}) {suite.path}")
    if not yes and not click.confirm("running?"):
        return

    echo.stdout_newline()
    # with Clients(config_inst) as client:
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
            if data_only:
                continue

            if not skip_jobs:
                try:
                    time_consuming = _run_pipeline_jobs(config_inst, suite, namespace, data_namespace_mangling)
                except Exception as e:
                    raise RuntimeError(f"exception occur while running pipeline jobs for {suite.path}") from e

            if not skip_data and clean_data:
                _delete_data(client, suite)
            echo.echo(f"[{i + 1}/{len(suites)}]elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
            if not skip_jobs:
                suite_file = str(suite.path).split("/")[-1]
                echo.echo(suite.pretty_final_summary(time_consuming, suite_file))

        except Exception:
            exception_id = uuid.uuid1()
            echo.echo(f"exception in {suite.path}, exception_id={exception_id}")
            LOGGER.exception(f"exception id: {exception_id}")
        finally:
            echo.stdout_newline()
    non_success_summary()
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


def _run_pipeline_jobs(config: Config, suite: Testsuite, namespace: str, data_namespace_mangling: bool):
    # pipeline demo goes here
    job_n = len(suite.pipeline_jobs)
    time_list = []
    for i, pipeline_job in enumerate(suite.pipeline_jobs):
        echo.echo(f"Running [{i + 1}/{job_n}] job: {pipeline_job.job_name}")

        def _raise(err_msg, status="failed"):
            exception_id = str(uuid.uuid1())
            suite.update_status(job_name=job_name, exception_id=exception_id, status=status)
            echo.file(f"exception({exception_id}), error message:\n{err_msg}")

        job_name, script_path = pipeline_job.job_name, pipeline_job.script_path
        mod = _load_module_from_script(script_path)
        start = time.time()
        try:
            if data_namespace_mangling:
                try:
                    mod.main(config=config, namespace=f"_{namespace}")
                    suite.update_status(job_name=job_name, status="success")
                    time_list.append(time.time() - start)

                except Exception as e:
                    _raise(e)
                    continue
            else:
                try:
                    mod.main(config=config)
                    suite.update_status(job_name=job_name, status="success")
                    time_list.append(time.time() - start)
                except Exception as e:
                    _raise(e)
                    continue
        except Exception as e:
            _raise(e, status="not submitted")
            continue

    return [str(int(i)) + "s" for i in time_list]
