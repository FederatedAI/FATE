import json
import typing
from pathlib import Path
from .uri_tools import parse_uri


def construct_local_dir(filepath: typing.Union[Path, str], *suffixes) -> "Path":
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    for suf in suffixes:
        filepath = filepath.joinpath(suf)

    return filepath


def write_json_file(path: str, buffer: dict):
    path = parse_uri(path).path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fout:
        fout.write(json.dumps(buffer, indent=2))
        fout.flush()


