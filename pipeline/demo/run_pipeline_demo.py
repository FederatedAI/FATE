import argparse
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

    conf = load_conf(args)
    new_config = _write_temp_conf(conf)

    path = Path(args.path)
    paths = _find_demo_files(path)
    # exclude demos
    if args.exclude is not None:
        exclude_paths = set()
        for p in args.exclude:
            exclude_paths.update(_find_demo_files(Path(p).resolve()))
        paths = [p for p in paths if p not in exclude_paths]

    # run demos
    summaries = run_demos(paths, new_config, summaries_base=Path(args.name).resolve())


def load_conf(args):
    file = args.config
    with open(file, "r") as f:
        conf = yaml.load(f, Loader=Loader)
    if args.back_end is not None:
        conf["backend"] = args.back_end
    if args.work_mode is not None:
        conf["work_mode"] = args.work_mode
    return conf

def _write_temp_conf(conf):

    # @TODO: write all config to temp file, then input temp file path to all demo
    # @TODO: record temp config in summaries text
    pass

def _clean_temp_conf(config):
    # @TODO: clean up temp config
    pass

def _find_demo_files(path):
    if path.is_file():
        if path.name.startswith("pipeline-") and path.name.endswith(".py"):
            paths = [path]
        else:
            LOGGER.warning(f"{path} is file, but does not start with `pipeline-` or is not a python file, skip")
            paths = []
    else:
        #@TODO: group demos by directory, if possible
        paths = path.glob(f"**/pipeline-*.py")
    return [p.resolve() for p in paths]

def run_demos(demos, config, summaries_base):
    #@TODO: run all demos with new specified config
    pass

if __name__ == "__main__":
    main()
