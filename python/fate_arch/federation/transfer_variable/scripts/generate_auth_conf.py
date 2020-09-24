import argparse
import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
from pathlib import Path

project_base = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(project_base.__str__())

from fate_arch.federation.transfer_variable import Variable, BaseTransferVariables

_camel_to_snake_pattern = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')


def _write(cls, base: Path):
    # noinspection PyProtectedMember
    def _find_all_variables(transfer_variable):
        d = {}
        for _k, _v in transfer_variable.__dict__.items():
            if isinstance(_v, Variable):
                d[_v._name] = _v
            elif isinstance(_v, BaseTransferVariables):
                d.update(_find_all_variables(_v))
        return d

    obj = cls()
    variables = {}
    module_and_class = f"{cls.__module__}.{cls.__name__}"
    for full_name, v in _find_all_variables(obj).items():
        if not isinstance(v, Variable):
            continue

        name_without_module = full_name[len(module_and_class) + 1:]
        # noinspection PyProtectedMember
        variables[name_without_module] = {"src": list(v._src), "dst": list(v._dst)}

    # empty
    if len(variables) <= 0:
        return

    auth_conf = {module_and_class: variables}

    base.mkdir(exist_ok=True)
    name = _camel_to_snake_pattern.sub(r'_\1', cls.__name__).lower()
    path = base.joinpath(f"{name}.json")
    old_conf = {}
    if path.exists():
        with path.open("r") as f:
            old_conf: dict = json.load(f)
    old_conf.update(auth_conf)

    # in order
    new_conf = {}
    for key in sorted(old_conf.keys()):
        new_conf[key] = old_conf[key]

    # save
    with path.open("w", ) as f:
        json.dump(new_conf, f, indent=2)


def _search_transfer_variables_class(path):
    module_name = path.name
    try:
        module_name = path.absolute().relative_to(project_base).with_suffix("").__str__().replace("/", ".")
        module = importlib.import_module(module_name)
        _class = inspect.getmembers(module, inspect.isclass)
        _subclass = [m for m in _class if issubclass(m[1], BaseTransferVariables) and m[1] != BaseTransferVariables]
        ret = [m[1] for m in _subclass if m[1].__module__ == module.__name__]
    except ImportError as e:
        print(f"import {module_name} fail, {e.args}, skip")
        raise e
    return ret


def main():
    arg_parser = argparse.ArgumentParser(description="convert transfer variables class to auth conf")
    arg_parser.add_argument("src", help="searching path or dir")
    arg_parser.add_argument("dst", help="dir to put auth conf")
    args = arg_parser.parse_args()

    cwd = os.getcwd()
    base = Path(cwd)
    search_base = base.joinpath(args.src).resolve()
    if search_base.is_file():
        files = [search_base]
    else:
        files = filter(lambda p: 'test' not in p.__str__() and "ftl" not in p.__str__(), search_base.glob("**/*.py"))
    dst_dir = base.joinpath(Path(args.dst)).resolve()
    #
    # input_text = input(f"[Y/N]convert class under: {search_base} to dir: {dst_dir}?")
    # if input_text.lower() != "y":
    #     return

    # noinspection PyProtectedMember
    Variable._disable_auth_check()

    for file_path in files:
        for cls in _search_transfer_variables_class(file_path):
            _write(cls, dst_dir)
    print("done")


if __name__ == '__main__':
    main()
