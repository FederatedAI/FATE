import time
import uuid
from datetime import timedelta
from inspect import signature

import click
from fate_test._client import Clients
from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test._parser import BenchmarkSuite
from fate_test.scripts._options import SharedOptions
from fate_test.scripts._utils import _upload_data, _delete_data, _load_testsuites, _load_module_from_script
from fate_test.utils import show_data, match_metrics


@click.command(name="benchmark-quality")
@click.option('-i', '--include', required=True, type=click.Path(exists=True), multiple=True, metavar="<include>",
              help="include *benchmark.json under these paths")
@click.option('-e', '--exclude', type=click.Path(exists=True), multiple=True,
              help="exclude *benchmark.json under these paths")
@click.option('-g', '--glob', type=str,
              help="glob string to filter sub-directory of path specified by <include>")
@click.option('-t', '--tol', type=float,
              help="tolerance (absolute error) for metrics to be considered almost equal. "
                   "Comparison is done by evaluating abs(a-b) <= max(relative_tol * max(abs(a), abs(b)), absolute_tol)")
@click.option('--skip-data', is_flag=True, default=False,
              help="skip uploading data specified in benchmark conf")
@click.option('--yes', is_flag=True,
              help="skip double check")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def run_benchmark(ctx, include, exclude, glob, skip_data, tol, yes, **kwargs):
    """
    process benchmark suite
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    data_namespace_mangling = ctx.obj["namespace_mangling"]

    echo.welcome("benchmark")
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
                if not skip_data:
                    try:
                        _upload_data(client, suite, config_inst)
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
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


@LOGGER.catch
def _run_benchmark_pairs(config: Config, suite: BenchmarkSuite, tol: float,
                         namespace: str, data_namespace_mangling: bool):
    # pipeline demo goes here
    pair_n = len(suite.pairs)
    for i, pair in enumerate(suite.pairs):
        echo.echo(f"Running [{i + 1}/{pair_n}] group: {pair.pair_name}")
        results = {}
        data_summary = None
        job_n = len(pair.jobs)
        for j, job in enumerate(pair.jobs):
            echo.echo(f"Running [{j + 1}/{job_n}] job: {job.job_name}")
            job_name, script_path, conf_path = job.job_name, job.script_path, job.conf_path
            param = Config.load_from_file(conf_path)
            mod = _load_module_from_script(script_path)
            input_params = signature(mod.main).parameters
            # local script
            if len(input_params) == 1:
                data, metric = mod.main(param=param)
            # pipeline script
            elif len(input_params) == 3:
                if data_namespace_mangling:
                    data, metric = mod.main(config=config, param=param, namespace=f"_{namespace}")
                else:
                    data, metric = mod.main(config=config, param=param)
            else:
                data, metric = mod.main()
            results[job_name] = metric
            if job_name == "FATE":
                data_summary = data
            if data_summary is None:
                data_summary = data
        rel_tol = pair.compare_setting.get("relative_tol")
        show_data(data_summary)
        match_metrics(evaluate=True, group_name=pair.pair_name, abs_tol=tol, rel_tol=rel_tol, **results)
