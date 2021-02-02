import os
import time
import uuid
from datetime import timedelta

import click
from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions
from fate_test.scripts import run_task


@click.group(name="flow_test")
def flow_group():
    """
    flow_test
    """
    ...


@flow_group.command("process")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def cli(ctx, **kwargs):
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

    def get_role(conf: Config):
        guest_party_id = conf.role['guest']
        host_party_id = conf.role['host']
        arbiter_party_id = conf.role['arbiter']
        online_serving = conf.serving_setting['serving_setting']['address']

        train_conf_path = os.path.abspath(conf.data_base_dir) + conf.unittest['train_conf_path']
        train_dsl_path = os.path.abspath(conf.data_base_dir) + conf.unittest['train_dsl_path']
        predict_conf_path = os.path.abspath(conf.data_base_dir) + conf.unittest['predict_conf_path']
        predict_dsl_path = os.path.abspath(conf.data_base_dir) + conf.unittest['predict_dsl_path']

        flow_services = conf.serving_setting['flow_services'][0]['address']
        run_task.server_url = "http://{}/{}".format(flow_services, run_task.API_VERSION)
        run_task.run_fate_flow_test(guest_party_id, host_party_id, arbiter_party_id, online_serving,
                                    train_conf_path, train_dsl_path, predict_conf_path, predict_dsl_path)

    try:
        start = time.time()
        get_role(config_inst)
        echo.echo(f"elapse {timedelta(seconds=int(time.time() - start))}", fg='red')
    except Exception:
        exception_id = uuid.uuid1()
        echo.echo(f"exception_id={exception_id}")
        LOGGER.exception(f"exception id: {exception_id}")
