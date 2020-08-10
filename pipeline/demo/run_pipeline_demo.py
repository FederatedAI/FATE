import argparse
import time
from pathlib import Path

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def main():
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("path", help="path to search pipeline-xxx.py")
    parser.add_argument("-config", default="./config.yaml", type=str,
                        help="config file")
    parser.add_argument("-name", default=f'pipeline-demo-{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    parser.add_argument("-work_mode", choices=[0, 1], type=int, help="work mode, if specified, overrides setting in config.yaml")
    parser.add_argument("-backend", choices=[0, 1], type=int, help="backend to use")
    parser.add_argument("-exclude", nargs="+", type=str)
    args = parser.parse_args()

    path = Path(args.path)
    conf = load_conf(args)



def load_conf(args):
    file = args.config
    with open(file, "r") as f:
        conf = yaml.load(f, Loader=Loader)
    if args.work_mode is not None:
        conf["work_mode"] = args.work_mode
    if args.back_end is not None:
        conf["back_end"] = args.back_end

    # @TODO: write all config to temp file, then input temp file path to all demo





if __name__ == "__main__":
    main()
