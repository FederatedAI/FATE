def disable_inner_logs():
    from ..common.log import getLogger

    logger = getLogger()
    logger.disabled = True
