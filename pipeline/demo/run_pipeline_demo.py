import argparse
import importlib
import tempfile
import time
from pathlib import Path

import loguru
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

LOGGER = loguru.logger


def main():
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("path", help="path to search pipeline-xxx.py")
    parser.add_argument("-config", default="./config.yaml", type=str,
                        help="config file")
    parser.add_argument("-name", default=f'pipeline-demo-{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    parser.add_argument("-backend", choices=[0, 1], type=int, help="backend to use")
    parser.add_argument("-work_mode", choices=[0, 1], type=int, help="work mode, if specified, overrides setting in config.yaml")
    parser.add_argument("-exclude", nargs="+", type=str)
    args = parser.parse_args()

    # find all demos
    path = Path(args.path)
    paths = _find_demo_files(path)

    # exclude demos
    if args.exclude is not None:
        exclude_paths = set()
        for p in args.exclude:
            exclude_paths.update(_find_demo_files(Path(p).resolve()))
        paths = [p for p in paths if p not in exclude_paths]

    # run demos
    conf = load_conf(args)
    summaries = run_demos(paths, conf, summaries_base=Path(args.name).resolve())

def load_conf(args):
    file = args.config
    with open(file, "r") as f:
        conf = yaml.load(f, Loader=Loader)
    if args.backend is not None:
        conf["backend"] = args.back_end
    if args.work_mode is not None:
        conf["work_mode"] = args.work_mode
    return conf

"""
def _write_temp_conf(conf):
    temp_config = tempfile.NamedTemporaryFile("rw", suffix='.yaml', delete=False)
    with temp_config as f:
        yaml.dump(conf, f, default_flow_style=False)
    return temp_config.name

def _clean_temp_conf(config):
    pass
"""

def _find_demo_files(path):
    if path.is_file():
        if path.name.startswith("pipeline-") and path.name.endswith(".py"):
            paths = [path]
        else:
            LOGGER.warning(f"{path} is file, but does not start with `pipeline-` or is not a python file, skip")
            paths = []
    else:
        # in future: group demos by directory
        paths = path.glob("**/pipeline-*.py")
    return [p.resolve() for p in paths]

def run_demos(demos, conf, summaries_base):
    temp_config = tempfile.NamedTemporaryFile("rw", suffix='.yaml')
    with temp_config as f:
        yaml.dump(conf, f, default_flow_style=False)
        for demo in demos:
            demo_module_path = ".".join(demo.split("/", -1)[:-1]).replace(".py", "")
            demo_module = importlib.import_module(demo_module_path)
            demo_module.main(temp_config)


if __name__ == "__main__":
    main()
