import logging.config
import rich.logging


def set_up_logging(rank, log_level="DEBUG"):
    if rank < 0:
        message_header = "[Main]"
    else:
        message_header = f"[Rank {rank}]"

    logging.config.dictConfig(
        dict(
            version=1,
            formatters={"with_rank": {"format": f"{message_header}%(message)s"}},
            handlers={
                "base": {
                    "class": "rich.logging.RichHandler",
                    "level": log_level,
                    "filters": [],
                    "formatter": "with_rank",
                    "tracebacks_show_locals": True,
                }
            },
            loggers={},
            root=dict(handlers=["base"], level=log_level),
            disable_existing_loggers=False,
        )
    )
    if rank < 0:
        from rich.traceback import install
        import click

        install(suppress=[click])
