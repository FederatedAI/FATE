import os
import uuid

import click
import json
from pathlib import Path
from ruamel import yaml
from fate_test._client import Clients
from fate_test._config import Config
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions
from fate_test.scripts.op_test.fate_he_performance_test import PaillierAssess


@click.group(name="secureprotol")
def secureprotol_group():
    """
    secureprotol test
    """
    ...


@secureprotol_group.command("paillier")
@click.option("-size", "--calls-count", type=int, help="", default=10000)
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def unit_test(ctx, calls_count, **kwargs):
    """
    paillier
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    if calls_count < 1000:
        calls_count = 1000

    if not yes and not click.confirm("running?"):
        return

    for method in ["Paillier"]:
        assess_table = PaillierAssess(method=method, calls_count=calls_count)
        table = assess_table.output_table()
        echo.echo(table)
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


@secureprotol_group.command("spdz")
@click.option("-calls", "--calls-count", type=int, help="", default=10000)
@click.option("-data", "--data-num", type=int, help="", default=1000)
@click.option("-seed", "--seed", type=int, help="", default=111)
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def spdz_test(ctx, calls_count, data_num, seed, **kwargs):
    """
    spdz
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    config_inst = ctx.obj["config"]
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
    if calls_count < 1000:
        calls_count = 1000

    if not yes and not click.confirm("running?"):
        return
    with Clients(config_inst) as client:
        start_both_sides(clients=client, conf=config_inst, calls_count=calls_count, data_num=data_num, seed=seed)
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


def start_both_sides(clients: Clients, conf: Config, calls_count, data_num, seed):
    def _update_param(parameters, key, value):
        if isinstance(parameters, dict):
            for keys in parameters:
                if keys == key:
                    parameters.get(key).update(value),
                elif isinstance(parameters[keys], dict):
                    _update_param(parameters[keys], key, value)

    def _get_config_file(file_dir: Path):
        with file_dir.open("r") as file:
            file_json = json.load(file)
        key = "spdz_secure_0"
        value = {'calls_count': calls_count, "data_num": data_num, 'seed': seed}
        _update_param(file_json, key, value)
        return file_json

    path = conf.flow_test_config_dir
    if isinstance(path, str):
        path = Path(path)
    config = {}
    if path is not None:
        with path.open("r") as f:
            config.update(yaml.safe_load(f))
    flow_test_template = config['flow_test_template']
    train_conf = _get_config_file(os.path.abspath(conf.data_base_dir) + flow_test_template['spdz_conf_path'])
    train_dsl = _get_config_file(os.path.abspath(conf.data_base_dir) + flow_test_template['spdz_dsl_path'])
    try:
        response = clients["guest_0"]._submit_job(conf=train_conf, dsl=train_dsl)
        job_id = response.job_id
        guest_party_id =  conf.role['guest']
        component_name = config['component_name']
        clients["guest_0"]._awaiting(job_id=job_id, role="guest")
        summary_file_dir = clients["guest_0"]._get_summary(job_id=job_id, role="guest", party_id=guest_party_id,
                                                           component_name=component_name)
        summary_file = _get_config_file(Path(summary_file_dir[0]))
        # 表格打印
    except Exception:
        exception_id = str(uuid.uuid1())
        echo.file(f"exception({exception_id})")
        LOGGER.exception(f"exception id: {exception_id}")
