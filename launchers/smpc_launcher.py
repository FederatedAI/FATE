import importlib

import click
from dataclasses import dataclass, field

from fate.arch.launchers.multiprocess_launcher import launch

from fate.arch.launchers.argparser import HfArgumentParser


@dataclass
class SMPCArguments:
    proc: str = field()


def run_mpc(ctx):
    from fate.ml.mpc import MPCModule

    # init mpc
    args, _ = HfArgumentParser(SMPCArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    ctx.mpc.init()

    # get proc cls
    module_name, cls_name = args.proc.split(":")
    module = importlib.import_module(module_name)
    mpc_module = getattr(module, cls_name)
    assert issubclass(mpc_module, MPCModule), f"{mpc_module} is not a subclass of MPCModule"
    inst = mpc_module()
    inst.fit(ctx)


if __name__ == "__main__":
    launch(run_mpc)
