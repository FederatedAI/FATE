import click


@click.command()
@click.option("--process-tag", required=True, help="unique id to identify this execution process")
@click.option("--config", required=False, type=click.File(), help="config path")
def cleanup(process_tag, config):
    import logging
    import traceback

    from fate.arch import Context
    from fate.components.core import load_computing, load_federation
    from fate.components.core.spec.task import TaskCleanupConfigSpec

    logger = logging.getLogger(__name__)
    config = TaskCleanupConfigSpec.parse_obj(config)

    try:
        computing = load_computing(config.computing)
        federation = load_federation(config.federation, computing)
        ctx = Context(
            computing=computing,
            federation=federation,
        )
        ctx.destroy()
    except Exception as e:
        traceback.print_exception(e)
        raise e


if __name__ == "__main__":
    cleanup()
