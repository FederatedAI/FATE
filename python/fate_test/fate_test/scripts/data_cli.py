import os
import re
import sys
import time
import uuid
import json
from datetime import timedelta

import click
import torchvision
from pathlib import Path
from ruamel import yaml

from fate_test import _config
from fate_test._config import Config
from fate_test._client import Clients
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions
from fate_test.scripts._utils import _upload_data, _load_testsuites, _delete_data, _big_data_task


@click.group(name="data")
def data_group():
    """
    upload or delete data in suite config files
    """
    ...


@data_group.command("upload")
@click.option('-i', '--include', required=False, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *benchmark.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *benchmark.json under these paths")
@click.option("-t", "--config-type", type=click.Choice(["min_test", "all_examples"]), default="min_test",
              help="config file")
@click.option('-g', '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option('-s', '--suite-type', required=False, type=click.Choice(["testsuite", "benchmark"]), default="testsuite",
              help="suite type")
@click.option('-r', '--role', type=str, default='all', help="role to process, default to `all`. "
                                                            "use option likes: `guest_0`, `host_0`, `host`")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def upload(ctx, include, exclude, glob, suite_type, role, config_type, **kwargs):
    """
    upload data defined in suite config files
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    config_inst.extend_sid = ctx.obj["extend_sid"]
    config_inst.auto_increasing_sid = ctx.obj["auto_increasing_sid"]
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    if len(include) != 0:
        echo.echo("loading testsuites:")
        suffix = "benchmark.json" if suite_type == "benchmark" else "testsuite.json"
        suites = _load_testsuites(includes=include, excludes=exclude, glob=glob,
                                  suffix=suffix, suite_type=suite_type)
        for suite in suites:
            if role != "all":
                suite.dataset = [d for d in suite.dataset if re.match(d.role_str, role)]
            echo.echo(f"\tdataset({len(suite.dataset)}) {suite.path}")
        if not yes and not click.confirm("running?"):
            return
        client_upload(suites=suites, config_inst=config_inst, namespace=namespace)
    else:
        config = get_config(config_inst)
        if config_type == 'min_test':
            config_file = config.min_test_data_config
        else:
            config_file = config.all_examples_data_config

        with open(config_file, 'r', encoding='utf-8') as f:
            upload_data = json.loads(f.read())

        echo.echo(f"\tdataset({len(upload_data['data'])}) {config_file}")
        if not yes and not click.confirm("running?"):
            return
        with Clients(config_inst) as client:
            data_upload(client, config_inst, upload_data)
        echo.farewell()
        echo.echo(f"testsuite namespace: {namespace}", fg='red')


@data_group.command("delete")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *benchmark.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *benchmark.json under these paths")
@click.option('-g', '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option('-s', '--suite-type', required=True, type=click.Choice(["testsuite", "benchmark"]), help="suite type")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def delete(ctx, include, exclude, glob, yes, suite_type, **kwargs):
    """
    delete data defined in suite config files
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    suffix = "benchmark.json" if suite_type == "benchmark" else "testsuite.json"

    suites = _load_testsuites(includes=include, excludes=exclude, glob=glob,
                              suffix=suffix, suite_type=suite_type)
    if not yes and not click.confirm("running?"):
        return

    for suite in suites:
        echo.echo(f"\tdataset({len(suite.dataset)}) {suite.path}")
    if not yes and not click.confirm("running?"):
        return
    with Clients(config_inst) as client:
        for i, suite in enumerate(suites):
            _delete_data(client, suite)
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


@data_group.command("generate")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *testsuite.json / *benchmark.json under these paths")
@click.option('-ht', '--host-data-type', default='tag_value', type=click.Choice(['dense', 'tag', 'tag_value']),
              help="Select the format of the host data")
@click.option('-p', '--encryption-type', type=click.Choice(['sha256', 'md5']),
              help="Entry ID encryption method for,  sha256 and md5")
@click.option('-m', '--match-rate', default=1.0, type=float,
              help="Intersection rate relative to guest, between [0, 1]")
@click.option('-s', '--sparsity', default=0.2, type=float,
              help="The sparsity of tag data, The value is between (0-1)")
@click.option('-ng', '--guest-data-size', type=int, default=10000,
              help="Set guest data set size, not less than 100")
@click.option('-nh', '--host-data-size', type=int,
              help="Set host data set size, not less than 100")
@click.option('-fg', '--guest-feature-num', type=int, default=20,
              help="Set guest feature dimensions")
@click.option('-fh', '--host-feature-num', type=int, default=200,
              help="Set host feature dimensions; the default is equal to the number of guest's size")
@click.option('-o', '--output-path', type=click.Path(exists=True),
              help="Customize the output path of generated data")
@click.option('--force', is_flag=True, default=False,
              help="Overwrite existing file")
@click.option('--split-host', is_flag=True, default=False,
              help="Divide the amount of host data equally among all the host tables in TestSuite")
@click.option('--upload-data', is_flag=True, default=False,
              help="Generated data will be uploaded")
@click.option('--remove-data', is_flag=True, default=False,
              help="The generated data will be deleted")
@click.option('--parallelize', is_flag=True, default=False,
              help="It is directly used to upload data, and will not generate data")
@click.option('--use-local-data', is_flag=True, default=False,
              help="The existing data of the server will be uploaded, This parameter is not recommended for "
                   "distributed applications")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def generate(ctx, include, host_data_type, encryption_type, match_rate, sparsity, guest_data_size,
             host_data_size, guest_feature_num, host_feature_num, output_path, force, split_host, upload_data,
             remove_data, use_local_data, parallelize, **kwargs):
    """
    create data defined in suite config files
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    config_inst.extend_sid = ctx.obj["extend_sid"]
    config_inst.auto_increasing_sid = ctx.obj["auto_increasing_sid"]
    if parallelize and upload_data:
        upload_data = False
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    if host_data_size is None:
        host_data_size = guest_data_size
    suites = _load_testsuites(includes=include, excludes=tuple(), glob=None)
    suites += _load_testsuites(includes=include, excludes=tuple(), glob=None,
                               suffix="benchmark.json", suite_type="benchmark")
    for suite in suites:
        if upload_data:
            echo.echo(f"\tdataget({len(suite.dataset)}) dataset({len(suite.dataset)}) {suite.path}")
        else:
            echo.echo(f"\tdataget({len(suite.dataset)}) {suite.path}")
    if not yes and not click.confirm("running?"):
        return

    _big_data_task(include, guest_data_size, host_data_size, guest_feature_num, host_feature_num, host_data_type,
                   config_inst, encryption_type, match_rate, sparsity, force, split_host, output_path, parallelize)
    if upload_data:
        if use_local_data:
            _config.use_local_data = 0
        _config.data_switch = remove_data
        client_upload(suites=suites, config_inst=config_inst, namespace=namespace, output_path=output_path)


@data_group.command("download")
@click.option("-t", "--type", type=click.Choice(["mnist"]), default="mnist",
              help="config file")
@click.option('-o', '--output-path', type=click.Path(exists=True),
              help="output path of mnist data, the default path is examples/data")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def download_mnists(ctx, output_path, **kwargs):
    """
    download mnist data for flow
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')

    if output_path is None:
        config = get_config(config_inst)
        output_path = str(config.data_base_dir) + "/examples/data/"
    if not yes and not click.confirm("running?"):
        return
    try:
        download_mnist(Path(output_path), "mnist_train")
        download_mnist(Path(output_path), "mnist_eval", is_train=False)
    except Exception:
        exception_id = uuid.uuid1()
        echo.echo(f"exception_id={exception_id}")
        LOGGER.exception(f"exception id: {exception_id}")
    finally:
        echo.stdout_newline()
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


@data_group.command("query_schema")
@click.option('-cpn', '--component-name', required=False, type=str, help="component name", default='dataio_0')
@click.option('-j', '--job-id', required=True, type=str, help="job id")
@click.option('-r', '--role', required=True, type=click.Choice(["guest", "host", "arbiter"]), help="job id")
@click.option('-p', '--party-id', required=True, type=str, help="party id")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def query_schema(ctx, component_name, job_id, role, party_id, **kwargs):
    """
    query the meta of the output data of a component
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]
    config_inst = ctx.obj["config"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')

    if not yes and not click.confirm("running?"):
        return
    with Clients(config_inst) as client:
        query_component_output_data(client, config_inst, component_name, job_id, role, party_id)
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


def get_config(conf: Config):
    return conf


def query_component_output_data(clients: Clients, config: Config, component_name, job_id, role, party_id):
    roles = config.role
    clients_role = None
    for k, v in roles.items():
        if int(party_id) in v and k == role:
            clients_role = role + "_" + str(v.index(int(party_id)))
    try:
        if clients_role is None:
            raise ValueError(f"party id {party_id} does not exist")

        try:
            table_info = clients[clients_role].output_data_table(job_id=job_id, role=role, party_id=party_id,
                                                                 component_name=component_name)
            table_info = clients[clients_role].table_info(table_name=table_info['name'],
                                                          namespace=table_info['namespace'])
        except Exception as e:
            raise RuntimeError(f"An exception occurred while getting data {clients_role}<-{component_name}") from e

        echo.echo("query_component_output_data result: {}".format(table_info))
        try:
            header = table_info['data']['schema']['header']
        except ValueError as e:
            raise ValueError(f"Obtain header from table error, error msg: {e}")

        result = []
        for idx, header_name in enumerate(header[1:]):
            result.append((idx, header_name))
        echo.echo("Queried header is {}".format(result))
    except Exception:
        exception_id = uuid.uuid1()
        echo.echo(f"exception_id={exception_id}")
        LOGGER.exception(f"exception id: {exception_id}")
    finally:
        echo.stdout_newline()


def download_mnist(base, name, is_train=True):
    dataset = torchvision.datasets.MNIST(
        root=base.joinpath(".cache"), train=is_train, download=True
    )
    converted_path = base.joinpath(name)
    converted_path.mkdir(exist_ok=True)

    inputs_path = converted_path.joinpath("images")
    inputs_path.mkdir(exist_ok=True)
    targets_path = converted_path.joinpath("targets")
    config_path = converted_path.joinpath("config.yaml")
    filenames_path = converted_path.joinpath("filenames")

    with filenames_path.open("w") as filenames:
        with targets_path.open("w") as targets:
            for idx, (img, target) in enumerate(dataset):
                filename = f"{idx:05d}"
                # save img
                img.save(inputs_path.joinpath(f"{filename}.jpg"))
                # save target
                targets.write(f"{filename},{target}\n")
                # save filenames
                filenames.write(f"{filename}\n")

    config = {
        "type": "vision",
        "inputs": {"type": "images", "ext": "jpg", "PIL_mode": "L"},
        "targets": {"type": "integer"},
    }
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, indent=2, default_flow_style=False)


def client_upload(suites, config_inst, namespace, output_path=None):
    with Clients(config_inst) as client:
        for i, suite in enumerate(suites):
            # noinspection PyBroadException
            try:
                echo.echo(f"[{i + 1}/{len(suites)}]start at {time.strftime('%Y-%m-%d %X')} {suite.path}", fg='red')
                try:
                    _upload_data(client, suite, config_inst, output_path)
                except Exception as e:
                    raise RuntimeError(f"exception occur while uploading data for {suite.path}") from e
            except Exception:
                exception_id = uuid.uuid1()
                echo.echo(f"exception in {suite.path}, exception_id={exception_id}")
                LOGGER.exception(f"exception id: {exception_id}")
            finally:
                echo.stdout_newline()
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


def data_upload(clients: Clients, conf: Config, upload_config):
    def _await_finish(job_id, task_name=None):
        deadline = time.time() + sys.maxsize
        start = time.time()
        param = dict(
            job_id=job_id,
            role=None
        )
        while True:
            stdout = clients["guest_0"].flow_client("job/query", param)
            status = stdout["data"][0]["f_status"]
            elapse_seconds = int(time.time() - start)
            date = time.strftime('%Y-%m-%d %X')
            if task_name:
                log_msg = f"[{date}][{task_name}]{status}, elapse: {timedelta(seconds=elapse_seconds)}"
            else:
                log_msg = f"[{date}]{job_id} {status}, elapse: {timedelta(seconds=elapse_seconds)}"
            if (status == "running" or status == "waiting") and time.time() < deadline:
                print(log_msg, end="\r")
                time.sleep(1)
                continue
            else:
                print(" " * 60, end="\r")  # clean line
                echo.echo(log_msg)
                return status

    task_data = upload_config["data"]
    for i, data in enumerate(task_data):
        format_msg = f"@{data['file']} >> {data['namespace']}.{data['table_name']}"
        echo.echo(f"[{time.strftime('%Y-%m-%d %X')}]uploading {format_msg}")
        try:
            data["file"] = str(os.path.join(conf.data_base_dir, data["file"]))
            param = dict(
                file=data["file"],
                head=data["head"],
                partition=data["partition"],
                table_name=data["table_name"],
                namespace=data["namespace"]
            )
            stdout = clients["guest_0"].flow_client("data/upload", param, drop=1)
            job_id = stdout.get('jobId', None)
            echo.echo(f"[{time.strftime('%Y-%m-%d %X')}]upload done {format_msg}, job_id={job_id}\n")
            if job_id is None:
                echo.echo("table already exist. To upload again, Please add '-f 1' in start cmd")
                continue
            _await_finish(job_id)
            param = dict(
                table_name=data["table_name"],
                namespace=data["namespace"]
            )
            stdout = clients["guest_0"].flow_client("table/info", param)

            count = stdout["data"]["count"]
            if count != data["count"]:
                raise AssertionError("Count of upload file is not as expect, count is: {},"
                                     "expect is: {}".format(count, data["count"]))
            echo.echo(f"[{time.strftime('%Y-%m-%d %X')}] check_data_out {stdout} \n")
        except Exception as e:
            exception_id = uuid.uuid1()
            echo.echo(f"exception in {data['file']}, exception_id={exception_id}")
            LOGGER.exception(f"exception id: {exception_id}")
            echo.echo(f"upload {i + 1}th data {data['table_name']} fail, exception_id: {exception_id}")
            # raise RuntimeError(f"exception occur while uploading data for {data['file']}") from e
        finally:
            echo.stdout_newline()
