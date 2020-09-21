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

import click
from pathlib import Path
import os
from ruamel import yaml

default_config = Path(__file__).parent.joinpath("config.yaml").resolve()


@click.group("Pipeline Config Tool")
def cli():
    pass


@click.command(name="config")
@click.option("-c", "--pipeline-conf-path", "config_path", type=click.Path(exists=True),
              help="Absolute path to pipeline configuration file.")
@click.option("-d", "--log-directory", type=click.Path(exists=True),
              help="Absolute path to pipeline logs directory.")
@click.option("--ip", type=click.STRING, help="Fate flow server ip address.")
@click.option("--port", type=click.INT, help="Fate flow server port.")
def _config(config_path, ip, port, log_directory):
    """
        \b
        - DESCRIPTION:
            Pipeline Config Command. User can choose to provide absolute path to conf file,
            or provide ip address and http port of a valid fate flow server. In addition,
            pipeline log directory can be optionally set to arbitrary location. Notice that,
            if both conf file and specifications are provided, the conf would be loaded in priority.
            In this case, other keywords would be ignored.

        \b
        - USAGE:
            pipeline config -c /data/projects/FATE/fate_client/pipeline/config.yaml
            pipeline config --ip 10.1.2.3 --port 9380 --log-directory /data/projects/FATE/fate_client/pipeline/logs
    """
    if config_path is None and (ip is None or port is None):
        print(
            {
                "retcode": 100,
                "retmsg": "Pipeline configuration failed. Please provides configuration file path "
                          "or server http ip address & port information & log directory."
            }
        )
        return

    if config_path is None:
        config_path = default_config

    with Path(config_path).open("r") as fin:
        config = yaml.safe_load(fin)

    if ip is not None:
        config["ip"] = ip
    if port is not None:
        config["port"] = port
    if log_directory is not None:
        config["log_directory"] = Path(log_directory).resolve().__str__()

    with default_config.open("w") as fout:
        yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)

    print(
        {
            "retcode": 0,
            "retmsg": "Pipeline configuration succeeded."
        }
    )


cli.add_command(_config)

if __name__ == '__main__':
    cli()
