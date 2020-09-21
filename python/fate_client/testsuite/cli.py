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
from pathlib import Path

import click

from testsuite._client import Clients
from testsuite._config import create_config, priority_config
from testsuite._flow_client import SubmitJobResponse, QueryJobResponse, JobProgress, DataProgress, \
    UploadDataResponse
from testsuite._io import set_logger, LOGGER, echo
from testsuite._parser import Testsuite, Config, DATA_JSON_HOOK, CONF_JSON_HOOK, DSL_JSON_HOOK, JSON_STRING


@click.group(name="cli")
def cli():
    ...


@cli.command(name="config")
@click.argument("cmd", type=click.Choice(["new", "show", "edit"], case_sensitive=False))
def _config(cmd):
    """
    new|show|edit testsuite config
    """
    if cmd == "new":
        create_config(Path("testsuite_config.yaml"))
        click.echo(f"create config file: testsuite_config.yaml")
    if cmd == "show":
        click.echo(f"priority config path is {priority_config()}")
    if cmd == "edit":
        click.edit(filename=priority_config())


@LOGGER.catch
@cli.command(name="suite")
@click.option('--namespace', default=time.strftime('%Y%m%d%H%M%S'), type=str,
              help="namespace to distinguish each run")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *testsuite.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *testsuite.json under these paths")
@click.option('-c', '--config', default=priority_config().__str__(), type=click.Path(exists=True),
              help=f"config path, defaults to {priority_config()}")
@click.option('-r', '--replace', default="{}", type=JSON_STRING,
              help="a json string represents mapping for replacing fields in data/conf/dsl")
@click.option("-g", '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option("--yes", is_flag=True,
              help="skip double check")
def run_suite(replace, namespace, config, include, exclude, glob, yes):
    """
    process testsuite
    """

    # prepare output dir and json hooks
    _prepare(namespace, replace)

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
                suite.reflash_configs(client.config)

                try:
                    _upload_data(client, suite)
                except Exception as e:
                    raise RuntimeError(f"exception occur while uploading data for {suite.path}") from e

                echo.stdout_newline()
                try:
                    _submit_job(client, suite)
                except Exception as e:
                    raise RuntimeError(f"exception occur while submit job for {suite.path}") from e

                _delete_data(client, suite)
                echo.echo(f"[{i + 1}/{len(suites)}]elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
                echo.echo(suite.pretty_final_summary(), fg='red')

            except Exception:
                exception_id = uuid.uuid1()
                echo.echo(f"exception in {suite.path}, exception_id={exception_id}")
                LOGGER.exception(f"exception id: {exception_id}")
            finally:
                echo.stdout_newline()

    echo.farewell()


def _parse_config(config):
    try:
        config_inst = Config.from_yaml(config)
    except Exception as e:
        raise RuntimeError(f"error parse config from {config}") from e
    return config_inst


def _prepare(namespace, replace):
    Path(f"logs/{namespace}").mkdir(exist_ok=True, parents=True)
    set_logger(f"logs/{namespace}/exception.log")
    echo.set_file(click.open_file(f'logs/{namespace}/stdout', "a"))

    DATA_JSON_HOOK.add_extend_namespace_hook(namespace)
    CONF_JSON_HOOK.add_extend_namespace_hook(namespace)
    DATA_JSON_HOOK.add_replace_hook(replace)
    CONF_JSON_HOOK.add_replace_hook(replace)
    DSL_JSON_HOOK.add_replace_hook(replace)


def _load_testsuites(includes, excludes, glob):
    def _find_testsuite_files(path):
        if isinstance(path, str):
            path = Path(path)
        if path.is_file():
            if path.name.endswith("testsuite.json"):
                paths = [path]
            else:
                LOGGER.warning(f"{path} is file, but not end with `testsuite.json`, skip")
                paths = []
        else:
            paths = path.glob(f"**/*testsuite.json")
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
        suites.append(Testsuite.load(suite_path.resolve()))
    return suites


def _upload_data(clients: Clients, suite: Testsuite):
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
                    suite.update_status(job_name=job.job_name, job_id=resp.job_id, status="submitted")

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


def main():
    cli()


if __name__ == '__main__':
    main()
