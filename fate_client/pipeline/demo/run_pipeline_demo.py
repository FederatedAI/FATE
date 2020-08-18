#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import argparse
import contextlib
import importlib
import tempfile
import time
from pathlib import Path
from pipeline.utils.logger import LOGGER, DEMO_LOG, LogFormat

import yaml
import sys


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


# LOGGER = loguru.logger

class StreamToLogger:
    def __init__(self, logger, level="INFO"):
        self._level = level
        self.logger = logger
        self.line_buffer = ''

    def write(self, buffer):
        temp_buffer = self.line_buffer + buffer
        self.log_buffer = ''
        for line in temp_buffer.splitlines(True):
            if line[-1] == '\r':
                self.logger.opt(depth=1).log(self._level, line.rstrip())
            else:
                self.line_buffer += line

    def flush(self):
        if self.line_buffer != '':
            self.logger.opt(depth=1).log(self._level, self.line_buffer.rstrip())
        self.line_buffer = ''


def main(args):
    demo_logger, demo_log_path = _add_logger(args.name)
    # find all demos
    path = Path(args.path)
    paths = _find_demo_files(path)

    # exclude demos
    if args.exclude is not None:
        exclude_paths = set()
        for p in args.exclude:
            exclude_paths.update(_find_demo_files(Path(p).resolve()))
        paths = [p for p in paths if p not in exclude_paths]

    # load conf, replace arg values if provided
    conf = load_conf(args)

    # run demos
    run_demos(paths, conf, path, demo_logger, demo_log_path)


def load_conf(args):
    file = args.config
    with open(file, "r") as f:
        conf = yaml.load(f, Loader=Loader)
    if args.backend is not None:
        conf["backend"] = args.back_end
    if args.work_mode is not None:
        conf["work_mode"] = args.work_mode
    return conf


def _add_logger(name):
    path = Path(name).joinpath("logs")
    if path.exists() and not path.is_dir():
        raise Exception(f"{name} exist, but is not a dir")
    if not path.exists():
        path.mkdir(parents=True)
    LOGGER.add(f"{path.joinpath('DEMO.log')}", level="INFO", colorize=True, format=LogFormat.SIMPLE)
    LOGGER.add(f"{path.joinpath('INFO.log')}", level="INFO", colorize=True, format=LogFormat.NORMAL)
    LOGGER.add(f"{path.joinpath('DEBUG.log')}", level="DEBUG", colorize=True, format=LogFormat.NORMAL)
    LOGGER.add(f"{path.joinpath('ERROR.log')}", level="ERROR", colorize=True, format=LogFormat.NORMAL)
    demo_logger = LOGGER.bind(log_type=DEMO_LOG)
    return demo_logger, path


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


def run_demos(demos, conf, path, demo_logger, demo_log_path):
    temp_config = tempfile.NamedTemporaryFile('w', suffix='.yaml')
    with temp_config as f:
        yaml.dump(conf, f, default_flow_style=False)
        for demo in demos:
            module_name = str(demo).split("/", -1)[-1].split(".")[0]
            loader = importlib.machinery.SourceFileLoader(module_name, str(demo))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            demo_module = importlib.util.module_from_spec(spec)
            loader.exec_module(demo_module)
            demo_logger.info(f"Running demo(s) in {path}.")
            demo_logger.info(f"Demo logs are located in {demo_log_path}.\n")
            demo_logger.info(f"Running demo {module_name}...")
            demo_module.main(temp_config.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RUN PIPELINE DEMO")
    parser.add_argument("path", help="path to search pipeline-xxx.py")
    parser.add_argument("-config", default="./config.yaml", type=str,
                        help="config file")
    parser.add_argument("-name", default=f'pipeline-demo-{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    parser.add_argument("-backend", choices=[0, 1], type=int, help="backend to use")
    parser.add_argument("-work_mode", choices=[0, 1], type=int,
                        help="work mode, if specified, overrides setting in config.yaml")
    parser.add_argument("-exclude", nargs="+", type=str)
    args = parser.parse_args()
    main(args)
