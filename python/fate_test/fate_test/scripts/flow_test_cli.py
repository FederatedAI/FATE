import os
import time
import uuid
import click
from datetime import timedelta
from pathlib import Path
from ruamel import yaml

from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions
from fate_test.flow_test import flow_rest_api, flow_sdk_api, flow_cli_api, flow_process


@click.group(name="flow-test")
def flow_group():
    """
    flow test
    """
    ...


@flow_group.command("process")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def process(ctx, **kwargs):
    """
    flow process test
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]

    echo.welcome("benchmark")
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    if not yes and not click.confirm("running?"):
        return
    try:
        start = time.time()
        flow_process.run_fate_flow_test(get_role(conf=config_inst))
        echo.echo(f"elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
    except Exception:
        exception_id = uuid.uuid1()
        echo.echo(f"exception_id={exception_id}")
        LOGGER.exception(f"exception id: {exception_id}")


@flow_group.command("rest")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def api(ctx, **kwargs):
    """
    flow rest api test
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]

    echo.welcome("benchmark")
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    if not yes and not click.confirm("running?"):
        return
    try:
        start = time.time()
        flow_rest_api.run_test_api(get_role(conf=config_inst))
        echo.echo(f"elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
    except Exception:
        exception_id = uuid.uuid1()
        echo.echo(f"exception_id={exception_id}")
        LOGGER.exception(f"exception id: {exception_id}")


@flow_group.command("sdk")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def api(ctx, **kwargs):
    """
    flow sdk api test
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]

    echo.welcome("benchmark")
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    if not yes and not click.confirm("running?"):
        return
    try:
        start = time.time()
        flow_sdk_api.run_test_api(get_role(conf=config_inst))
        echo.echo(f"elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
    except Exception:
        exception_id = uuid.uuid1()
        echo.echo(f"exception_id={exception_id}")
        LOGGER.exception(f"exception id: {exception_id}")


@flow_group.command("cli")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def api(ctx, **kwargs):
    """
    flow cli api test
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]

    echo.welcome("benchmark")
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    echo.echo("loading testsuites:")
    if not yes and not click.confirm("running?"):
        return
    try:
        start = time.time()
        flow_cli_api.run_test_api(get_role(conf=config_inst))
        echo.echo(f"elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
    except Exception:
        exception_id = uuid.uuid1()
        echo.echo(f"exception_id={exception_id}")
        LOGGER.exception(f"exception id: {exception_id}")


def get_role(conf: Config):
    flow_services = conf.serving_setting['flow_services'][0]['address']
    path = conf.flow_test_config_dir
    if isinstance(path, str):
        path = Path(path)
    config = {}
    if path is not None:
        with path.open("r") as f:
            config.update(yaml.safe_load(f))
    flow_test_template = config['flow_test_template']
    config_json = {'guest_party_id': conf.role['guest'],
                   'host_party_id': [conf.role['host'][0]],
                   'arbiter_party_id': conf.role['arbiter'],
                   'online_serving': conf.serving_setting['serving_setting']['address'],
                   'work_mode': conf.work_mode,
                   'train_conf_path': os.path.abspath(conf.data_base_dir) + flow_test_template['train_conf_path'],
                   'train_dsl_path': os.path.abspath(conf.data_base_dir) + flow_test_template['train_dsl_path'],
                   'predict_conf_path': os.path.abspath(conf.data_base_dir) + flow_test_template['predict_conf_path'],
                   'predict_dsl_path': os.path.abspath(conf.data_base_dir) + flow_test_template['predict_dsl_path'],
                   'upload_file_path': os.path.abspath(conf.data_base_dir) + flow_test_template['upload_conf_path'],
                   'server_url': "http://{}/{}".format(flow_services, config['api_version']),
                   'train_auc': config['train_auc'],
                   'component_name': config['component_name'],
                   'metric_output_path': config['metric_output_path'],
                   'model_output_path': config['model_output_path'],
                   'cache_directory': conf.cache_directory,
                   'data_base_dir': conf.data_base_dir
                   }
    return config_json
