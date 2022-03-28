import click
import os
from fate_test._io import LOGGER, echo
from fate_test.scripts._options import SharedOptions
from fate_test.scripts.op_test.fate_he_performance_test import PaillierAssess
from fate_test.scripts.op_test.spdz_test import SPDZTest


@click.group(name="secure_protocol")
def secure_protocol_group():
    """
    secureprotol test
    """
    ...


@secure_protocol_group.command("paillier")
@click.option("-round", "--test-round", type=int, help="", default=1)
@click.option("-num", "--data-num", type=int, help="", default=10000)
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def paillier_test(ctx, data_num, test_round, **kwargs):
    """
    paillier
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')

    if not yes and not click.confirm("running?"):
        return

    for method in ["Paillier"]:
        assess_table = PaillierAssess(method=method, data_num=data_num, test_round=test_round)
        table = assess_table.output_table()
        echo.echo(table)
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')


@secure_protocol_group.command("spdz")
@click.option("-round", "--test-round", type=int, help="", default=1)
@click.option("-num", "--data-num", type=int, help="", default=10000)
@click.option("-partition", "--data-partition", type=int, help="", default=4)
@click.option("-lower_bound", "--data-lower-bound", type=int, help="", default=-1e9)
@click.option("-upper_bound", "--data-upper-bound", type=int, help="", default=1e9)
@click.option("-seed", "--seed", type=int, help="", default=123)
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def spdz_test(ctx, data_num, seed, data_partition, test_round,
              data_lower_bound, data_upper_bound, **kwargs):
    """
    spdz_test
    """
    ctx.obj.update(**kwargs)
    ctx.obj.post_process()
    namespace = ctx.obj["namespace"]
    yes = ctx.obj["yes"]
    echo.welcome()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')

    if not yes and not click.confirm("running?"):
        return

    conf = ctx.obj["config"]
    runtime_config_path_prefix = \
        os.path.abspath(conf.fate_base) + "/python/fate_test/fate_test/scripts/op_test/spdz_conf/"

    params = dict(data_num=data_num, seed=seed, data_partition=data_partition,
                  test_round=test_round, data_lower_bound=data_lower_bound,
                  data_upper_bound=data_upper_bound)

    flow_address = None
    for idx, address in enumerate(conf.serving_setting["flow_services"]):
        if conf.role["guest"][0] in address["parties"]:
            flow_address = address["address"]

    spdz_test = SPDZTest(params=params,
                         conf_path=runtime_config_path_prefix + "job_conf.json",
                         dsl_path=runtime_config_path_prefix + "job_dsl.json",
                         flow_address=flow_address,
                         guest_party_id=[conf.role["guest"][0]],
                         host_party_id=[conf.role["host"][0]])

    tables = spdz_test.run()
    for table in tables:
        echo.echo(table)
    echo.farewell()
    echo.echo(f"testsuite namespace: {namespace}", fg='red')
