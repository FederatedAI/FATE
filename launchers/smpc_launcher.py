import importlib

import click

from fate.arch.launchers.multiprocess_launcher import launch


def run_mpc(ctx, proc, parameters):
    from fate.ml.mpc import MPCModule

    # init mpc

    ctx.mpc.init()

    # get proc cls
    module_name, cls_name = proc.split(":")
    module = importlib.import_module(module_name)
    mpc_module = getattr(module, cls_name)
    assert issubclass(mpc_module, MPCModule), f"{mpc_module} is not a subclass of MPCModule"
    parameters = mpc_module.parse_parameters(parameters)
    inst = mpc_module(**parameters)
    inst.fit(ctx)


@click.command()
@click.option("--federation_session_id", type=str, help="federation session id")
@click.option("--parties", multiple=True, type=str, help="parties", required=True)
@click.option("--data_dir", type=str, help="data dir")
@click.option("--proc", type=str, help="proc, e.g. fate.ml.mpc.svm:SVM", required=True)
@click.option("--log_level", type=str, help="log level", default="INFO")
@click.option("-p", "--parameter", multiple=True, type=str, help="parameters")
def cli(federation_session_id, parties, data_dir, proc, log_level, parameter):
    launch(run_mpc, federation_session_id, parties, data_dir, proc, log_level, parameter)


if __name__ == "__main__":
    cli()
