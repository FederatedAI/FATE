import typing
import concurrent.futures


_remote_futures = set()


def _clear_callback(future):
    _remote_futures.remove(future)


def add_remote_futures(fs: typing.List[concurrent.futures.Future]):
    for f in fs:
        f.add_done_callback(_clear_callback)
        _remote_futures.add(f)


def wait_all_remote_done(timeout=None):
    concurrent.futures.wait(
        _remote_futures, timeout=timeout, return_when=concurrent.futures.ALL_COMPLETED
    )
