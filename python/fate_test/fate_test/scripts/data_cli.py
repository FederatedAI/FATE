import re
import time
import uuid

import click
from fate_test import _config
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
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *benchmark.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *benchmark.json under these paths")
@click.option('-g', '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option('-s', '--suite-type', required=True, type=click.Choice(["testsuite", "benchmark"]), help="suite type")
@click.option('-r', '--role', type=str, default='all', help="role to process, default to `all`. "
                                                            "use option likes: `guest_0`, `host_0`, `host`")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def upload(ctx, include, exclude, glob, suite_type, role, **kwargs):
    """
    upload data defined in suite config files
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
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
@click.option('--use-local-data', is_flag=True, default=False,
              help="The existing data of the server will be uploaded, This parameter is not recommended for "
                   "distributed applications")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def generate(ctx, include, host_data_type, encryption_type, match_rate, sparsity, guest_data_size,
             host_data_size, guest_feature_num, host_feature_num, output_path, force, split_host, upload_data,
             remove_data, use_local_data, **kwargs):
    """
    create data defined in suite config files
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    if host_data_size is None:
        host_data_size = guest_data_size
    suites = _load_testsuites(includes=include, excludes=tuple(), glob=None)
    suites += _load_testsuites(includes=include, excludes=tuple(), glob=None, suffix="benchmark.json", suite_type="benchmark")
    for suite in suites:
        if upload_data:
            echo.echo(f"\tdataget({len(suite.dataset)}) dataset({len(suite.dataset)}) {suite.path}")
        else:
            echo.echo(f"\tdataget({len(suite.dataset)}) {suite.path}")
    if not yes and not click.confirm("running?"):
        return

    _big_data_task(include, guest_data_size, host_data_size, guest_feature_num, host_feature_num, host_data_type,
                   config_inst, encryption_type, match_rate, sparsity, force, split_host, output_path)
    if upload_data:
        if use_local_data:
            _config.use_local_data = 0
        _config.data_switch = remove_data
        client_upload(suites=suites, config_inst=config_inst, namespace=namespace, output_path=output_path)


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
