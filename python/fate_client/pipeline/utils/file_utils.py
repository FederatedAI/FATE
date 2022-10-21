import json
import os.path
from pathlib import Path


def construct_local_dir(path, default_suffix=None):
    if not path:
        ret_dir = Path.cwd().joinpath("data") # .resolve().as_uri()
        if default_suffix:
            for suf in default_suffix:
                ret_dir = ret_dir.joinpath(suf)
        ret_dir = ret_dir
    else:
        ret_dir = Path(path)

    return ret_dir


def generate_dir_uri(path: Path, *suffixes):
    for suf in suffixes:
        path = path.joinpath(suf)

    return path.resolve().as_uri()


def generate_dir(path: Path, *suffixes) -> "Path":
    for suf in suffixes:
        path = path.joinpath(str(suf))

    return path.resolve()


def write_json_file(path, buffer):
    create_parent_dir(path)
    with open(path, "w") as fout:
        fout.write(json.dumps(buffer))
        fout.flush()


def create_parent_dir(path):
    parent_dir = Path(path).parent
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
