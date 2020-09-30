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
import importlib
from datetime import timedelta
from inspect import signature
from pathlib import Path

import click

from fate_test._client import Clients
from fate_test._config import create_config, priority_config
from fate_test._flow_client import SubmitJobResponse, QueryJobResponse, JobProgress, DataProgress, \
    UploadDataResponse
from fate_test._io import set_logger, LOGGER, echo
from fate_test._parser import Testsuite, BenchmarkSuite, Config, DATA_JSON_HOOK, CONF_JSON_HOOK, DSL_JSON_HOOK, \
    JSON_STRING
from fate_test.utils import match_metrics


@click.group(name="cli")
def cli():
    ...


@cli.command(name="config")
@click.argument("cmd", type=click.Choice(["new", "show", "edit", "check"], case_sensitive=False))
@click.option('-r', '--role', required=False, type=str)
def _config(cmd, role):
    """
    new|show|edit fate test config
    """
    if cmd == "new":
        create_config(Path("fate_test_config.yaml"))
        click.echo(f"create config file: fate_test_config.yaml")
    if cmd == "show":
        click.echo(f"priority config path is {priority_config()}")
    if cmd == "edit":
        click.edit(filename=priority_config())
    if cmd == "check":
        if not role:
            click.echo("use --role to specify role to check, "
                       "such as --role guest_0 to check 0th guest, "
                       "or --role all to check all roles in config")
            return
        config_inst = _parse_config(priority_config())
        with Clients(config_inst) as clients:
            if role != "all" and not clients.contains(role):
                click.echo(f"[X]{role} not in config")
                return
            roles = clients.all_roles() if role == "all" else [role]
            for r in roles:
                try:
                    version, address = clients[r].check_connection()
                except Exception as e:
                    click.echo(f"[X]connection {address} fail, role is {r}, exception is {e.args}")
                click.echo(f"[✓]connection {address} ok, fate version is {version}, role is {r}")


@LOGGER.catch
@cli.command(name="suite")
@click.option('--data-namespace-mangling', type=bool, is_flag=True, default=False,
              help="mangling data namespace")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *testsuite.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *testsuite.json under these paths")
@click.option('-c', '--config', default=priority_config().__str__(), type=click.Path(exists=True),
              help=f"specify config path")
@click.option('-r', '--replace', default="{}", type=JSON_STRING,
              help="a json string represents mapping for replacing fields in data/conf/dsl")
@click.option("-g", '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option("--yes", is_flag=True,
              help="skip double check")
@click.option("--skip-dsl-jobs", is_flag=True, default=False,
              help="skip dsl jobs defined in testsuite")
@click.option("--skip-pipeline-jobs", is_flag=True, default=False,
              help="skip pipeline jobs defined in testsuite")
@click.option("--skip-data", is_flag=True, default=False,
              help="skip uploading data specified in testsuite")
def run_suite(replace, data_namespace_mangling, config, include, exclude, glob,
              skip_dsl_jobs, skip_pipeline_jobs, skip_data, yes):
    """
    process testsuite
    """
    namespace = time.strftime('%Y%m%d%H%M%S')
    # prepare output dir and json hooks
    _prepare(data_namespace_mangling, namespace, replace)

    echo.welcome()
    config_inst = _parse_config(config)
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    suites = _load_testsuites(includes=include, excludes=exclude, glob=glob)
    for suite in suites:
        echo.echo(f"\tdataset({len(suite.dataset)}) jobs({len(suite.jobs)}) {suite.path}")
    if not yes and not click.confirm("running?"):
        return

    echo.stdout_newline()

    with Clients(config_inst) as client:
        for i, suite in enumerate(suites):
            # noinspection PyBroadException
            try:
                start = time.time()
                echo.echo(f"[{i + 1}/{len(suites)}]start at {time.strftime('%Y-%m-%d %X')} {suite.path}", fg='red')
                suite.reflash_configs(config_inst)

                if not skip_data:
                    try:
                        _upload_data(client, suite)
                    except Exception as e:
                        raise RuntimeError(f"exception occur while uploading data for {suite.path}") from e

                if not skip_dsl_jobs:
                    echo.stdout_newline()
                    try:
                        _submit_job(client, suite)
                    except Exception as e:
                        raise RuntimeError(f"exception occur while submit job for {suite.path}") from e

                if not skip_pipeline_jobs:
                    try:
                        _run_pipeline_jobs(config_inst, suite, namespace, data_namespace_mangling)
                    except Exception as e:
                        raise RuntimeError(f"exception occur while running pipeline jobs for {suite.path}") from e

                if not skip_data:
                    _delete_data(client, suite)
                echo.echo(f"[{i + 1}/{len(suites)}]elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
                if not skip_dsl_jobs:
                    echo.echo(suite.pretty_final_summary(), fg='red')

            except Exception:
                exception_id = uuid.uuid1()
                echo.echo(f"exception in {suite.path}, exception_id={exception_id}")
                LOGGER.exception(f"exception id: {exception_id}")
            finally:
                echo.stdout_newline()

    echo.farewell()


@LOGGER.catch
@cli.command(name="benchmark-quality")
@click.option('--data-namespace-mangling', type=bool, is_flag=True, default=False,
              help="mangling data namespace")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *benchmark.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *benchmark.json under these paths")
@click.option('-c', '--config', default=priority_config().__str__(), type=click.Path(exists=True),
              help=f"specify config path")
@click.option('-g', '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option('-t', '--tol', type=float,
              help="tolerance (absolute error) for metrics to be considered almost equal. "
                   "Comparison is done by evaluating abs(a-b) <= max(relative_tol * max(abs(a), abs(b)), absolute_tol)")
@click.option('--yes', is_flag=True,
              help="skip double check")
@click.option('--skip-data', is_flag=True, default=False,
              help="skip uploading data specified in benchmark conf")
def run_benchmark(data_namespace_mangling, config, include, exclude, glob, skip_data, tol, yes):
    """
    process benchmark suite
    """
    namespace = time.strftime('%Y%m%d%H%M%S')
    # prepare output dir and json hooks
    _prepare(data_namespace_mangling, namespace, replace={})

    echo.welcome("benchmark")
    config_inst = _parse_config(config)
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    suites = _load_testsuites(includes=include, excludes=exclude, glob=glob,
                              suffix="benchmark.json", suite_type="benchmark")
    for suite in suites:
        echo.echo(f"\tdataset({len(suite.dataset)}) benchmark groups({len(suite.pairs)}) {suite.path}")
    if not yes and not click.confirm("running?"):
        return
    with Clients(config_inst) as client:
        for i, suite in enumerate(suites):
            # noinspection PyBroadException
            try:
                start = time.time()
                echo.echo(f"[{i + 1}/{len(suites)}]start at {time.strftime('%Y-%m-%d %X')} {suite.path}", fg='red')
                suite.reflash_configs(config_inst)

                if not skip_data:
                    try:
                        _upload_data(client, suite)
                    except Exception as e:
                        raise RuntimeError(f"exception occur while uploading data for {suite.path}") from e
                try:
                    _run_benchmark_pairs(config_inst, suite, tol, namespace, data_namespace_mangling)
                except Exception as e:
                    raise RuntimeError(f"exception occur while running benchmark jobs for {suite.path}") from e

                if not skip_data:
                    _delete_data(client, suite)
                echo.echo(f"[{i + 1}/{len(suites)}]elapse {timedelta(seconds=int(time.time() - start))}", fg='red')

            except Exception:
                exception_id = uuid.uuid1()
                echo.echo(f"exception in {suite.path}, exception_id={exception_id}")
                LOGGER.exception(f"exception id: {exception_id}")
            finally:
                echo.stdout_newline()
    echo.farewell()


def _parse_config(config):
    try:
        config_inst = Config.load(config)
    except Exception as e:
        raise RuntimeError(f"error parse config from {config}") from e
    return config_inst


def _prepare(data_namespace_mangling, namespace, replace):
    Path(f"logs/{namespace}").mkdir(exist_ok=True, parents=True)
    set_logger(f"logs/{namespace}/exception.log")
    echo.set_file(click.open_file(f'logs/{namespace}/stdout', "a"))

    if data_namespace_mangling:
        echo.echo(f"add data_namespace_mangling: _{namespace}")
        DATA_JSON_HOOK.add_extend_namespace_hook(namespace)
        CONF_JSON_HOOK.add_extend_namespace_hook(namespace)
    DATA_JSON_HOOK.add_replace_hook(replace)
    CONF_JSON_HOOK.add_replace_hook(replace)
    DSL_JSON_HOOK.add_replace_hook(replace)


def _load_testsuites(includes, excludes, glob, suffix="testsuite.json", suite_type="testsuite"):
    def _find_testsuite_files(path):
        if isinstance(path, str):
            path = Path(path)
        if path.is_file():
            if path.name.endswith(suffix):
                paths = [path]
            else:
                LOGGER.warning(f"{path} is file, but not end with `{suffix}`, skip")
                paths = []
        else:
            paths = path.glob(f"**/*{suffix}")
        return [p.resolve() for p in paths]

    excludes_set = set()
    for exclude in excludes:
        excludes_set.update(_find_testsuite_files(exclude))

    suite_paths = set()
    for include in includes:
        if isinstance(include, str):
            include = Path(include)

        # glob
        if glob is not None and include.is_dir():
            include_list = include.glob(glob)
        else:
            include_list = [include]
        for include_path in include_list:
            for suite_path in _find_testsuite_files(include_path):
                if suite_path not in excludes_set:
                    suite_paths.add(suite_path)
    suites = []
    for suite_path in suite_paths:
        if suite_type == "testsuite":
            suites.append(Testsuite.load(suite_path.resolve()))
        elif suite_type == "benchmark":
            suites.append(BenchmarkSuite.load(suite_path.resolve()))
        else:
            raise ValueError(f"Unsupported suite type: {suite_type}. Only accept type 'testsuite' or 'benchmark'.")
    return suites


def _upload_data(clients: Clients, suite):
    with click.progressbar(length=len(suite.dataset),
                           label="dataset",
                           show_eta=False,
                           show_pos=True,
                           width=24) as bar:
        for i, data in enumerate(suite.dataset):
            data_progress = DataProgress(data.role_str)

            def update_bar(n_step):
                bar.item_show_func = lambda x: data_progress.show()
                time.sleep(0.1)
                bar.update(n_step)

            def _call_back(resp):
                if isinstance(resp, UploadDataResponse):
                    data_progress.submitted(resp.job_id)
                    echo.file(f"[dataset]{resp.job_id}")
                if isinstance(resp, QueryJobResponse):
                    data_progress.update()
                update_bar(0)

            try:
                echo.stdout_newline()
                response = clients[data.role_str].upload_data(data, _call_back)
                data_progress.update()
                if not response.status.is_success():
                    raise RuntimeError(f"uploading {i + 1}th data for {suite.path} {response.status}")
                bar.update(1)
            except Exception as e:
                raise RuntimeError(f"exception uploading {i + 1}th data") from e


def _delete_data(clients: Clients, suite: Testsuite):
    with click.progressbar(length=len(suite.dataset),
                           label="delete ",
                           show_eta=False,
                           show_pos=True,
                           width=24) as bar:
        for data in suite.dataset:
            # noinspection PyBroadException
            try:
                bar.item_show_func = \
                    lambda x: f"delete table: name={data.config['table_name']}, namespace={data.config['namespace']}"
                clients[data.role_str].delete_data(data)
            except Exception:
                LOGGER.exception(
                    f"delete failed: name={data.config['table_name']}, namespace={data.config['namespace']}")

            time.sleep(0.5)
            bar.update(1)
            echo.stdout_newline()


def _submit_job(clients: Clients, suite: Testsuite):
    # submit jobs
    with click.progressbar(length=len(suite.jobs),
                           label="jobs   ",
                           show_eta=False,
                           show_pos=True,
                           width=24) as bar:
        for job in suite.jobs_iter():

            job_progress = JobProgress(job.job_name)

            def update_bar(n_step):
                bar.item_show_func = lambda x: job_progress.show()
                time.sleep(0.5)
                bar.update(n_step)

            update_bar(1)

            def _call_back(resp: SubmitJobResponse):
                if isinstance(resp, SubmitJobResponse):
                    job_progress.submitted(resp.job_id)
                    echo.file(f"[jobs] {resp.job_id} ", nl=False)
                    suite.update_status(job_name=job.job_name, job_id=resp.job_id)

                if isinstance(resp, QueryJobResponse):
                    job_progress.running(resp.status, resp.progress)

                update_bar(0)

            # noinspection PyBroadException
            try:
                response = clients["guest_0"].submit_job(job, _call_back)
            except Exception:
                exception_id = str(uuid.uuid1())
                job_progress.exception(exception_id)
                suite.update_status(job_name=job.job_name, exception_id=exception_id)
                echo.file(f"exception({exception_id})")
                LOGGER.exception(f"exception id: {exception_id}")
            else:
                job_progress.final(response.status)
                suite.update_status(job_name=job.job_name, status=response.status.status)
                if response.status.is_success():
                    suite.feed_success_model_info(job.job_name, response.model_info)
            update_bar(0)
            echo.stdout_newline()


def _load_module_from_script(script_path):
    module_name = str(script_path).split("/", -1)[-1].split(".")[0]
    loader = importlib.machinery.SourceFileLoader(module_name, str(script_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _run_pipeline_jobs(config: Config, suite: Testsuite, namespace: str, data_namespace_mangling: bool):
    # pipeline demo goes here
    for pipeline_job in suite.pipeline_jobs:
        job_name, script_path = pipeline_job.job_name, pipeline_job.script_path
        mod = _load_module_from_script(script_path)
        if data_namespace_mangling:
            mod.main(config, f"_{namespace}")
        else:
            mod.main(config)


def _run_benchmark_pairs(config: Config, suite: BenchmarkSuite, tol: float,
                         namespace: str, data_namespace_mangling: bool):
    # pipeline demo goes here
    pair_n = len(suite.pairs)
    for i, pair in enumerate(suite.pairs):
        echo.echo(f"Running {i + 1} of {pair_n} groups: {pair.pair_name}")
        results = {}
        job_n = len(pair.jobs)
        for job in pair.jobs:
            echo.echo(f"Running {i + 1} of {job_n} jobs: {job.job_name}")
            job_name, script_path, conf_path = job.job_name, job.script_path, job.conf_path
            param = Config.load_from_file(conf_path)
            mod = _load_module_from_script(script_path)
            input_params = signature(mod.main).parameters
            # local script
            if len(input_params) == 1:
                metric = mod.main(param=param)
            # pipeline script
            elif len(input_params) == 3:
                if data_namespace_mangling:
                    metric = mod.main(config=config, param=param, namespace=f"_{namespace}")
                else:
                    metric = mod.main(config=config, param=param)
            else:
                metric = mod.main()
            results[job_name] = metric
        rel_tol = pair.compare_setting.get("relative_tol")
        match_metrics(evaluate=True, abs_tol=tol, rel_tol=rel_tol, **results)


def main():
    cli()


if __name__ == '__main__':
    main()
