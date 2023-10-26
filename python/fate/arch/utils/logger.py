import logging.config


def set_up_logging(rank, log_level="DEBUG"):
    logging.config.dictConfig(
        dict(
            version=1,
            formatters={"with_rank": {"format": f"[Rank {rank}]%(message)s"}},
            handlers={
                "base": {
                    "class": "rich.logging.RichHandler",
                    "level": log_level,
                    "filters": [],
                    "formatter": "with_rank",
                }
            },
            loggers={},
            root=dict(handlers=["base"], level=log_level),
            disable_existing_loggers=False,
        )
    )
