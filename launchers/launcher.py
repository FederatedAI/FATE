import click
import importlib


@click.command()
@click.option("--csession_id", type=str, help="computing session id")
@click.option("--federation_session_id", type=str, help="federation session id", required=True)
@click.option("--rank", type=int, help="rank", required=True)
@click.option("--parties", multiple=True, type=str, help="parties", required=True)
@click.option("--data_dir", type=str, help="data dir")
@click.option("--proc", type=str, help="proc, e.g. fate.ml.mpc.svm:SVM", required=True)
def cli(csession_id, federation_session_id, rank, parties, data_dir, proc):
    from fate.arch.utils.logger import set_up_logging
    from fate.arch.utils.context_helper import init_standalone_context

    # set up logging
    set_up_logging(rank)

    # init context
    parties = [tuple(p.split(":")) for p in parties]
    if rank >= len(parties):
        raise ValueError(f"rank {rank} is out of range {len(parties)}")
    party = parties[rank]
    if not csession_id:
        csession_id = f"{federation_session_id}_{party[0]}_{party[1]}"
    ctx = init_standalone_context(csession_id, federation_session_id, party, parties, data_dir)

    # init crypten
    from fate.ml.mpc import MPCModule

    ctx.mpc.init()

    # get proc cls
    module_name, cls_name = proc.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    assert issubclass(cls, MPCModule), f"{cls} is not a subclass of MPCModule"

    inst = cls()
    inst.fit(ctx)


if __name__ == "__main__":
    cli()
