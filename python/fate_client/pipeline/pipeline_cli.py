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
from ruamel import yaml

default_config = Path(__file__).parent.joinpath("config.yaml").resolve()


@click.group("Pipeline Config Tool")
def cli():
    pass


@click.command(name="init")
@click.option("-c", "--pipeline-conf-path", "config_path", type=click.Path(exists=True),
              help="Path to pipeline configuration file.")
@click.option("-d", "--log-directory", type=click.Path(),
              help="Path to pipeline logs directory.")
@click.option("--ip", type=click.STRING, help="Fate flow server ip address.")
@click.option("--port", type=click.INT, help="Fate flow server port.")
@click.option("-r", "--system-user", type=click.STRING, help="system user role")
def _init(**kwargs):
    """
        \b
        - DESCRIPTION:
            Pipeline Config Command. User can choose to provide path to conf file,
            or provide ip address and http port of a valid fate flow server. Optionally,
            pipeline log directory can be set to arbitrary location. Default log directory is
            pipeline/logs. Notice that, if both conf file and specifications are provided,
            settings in conf file are ignored.

        \b
        - USAGE:
            pipeline init -c config.yaml
            pipeline init --ip 10.1.2.3 --port 9380 --log-directory ./logs --system-user guest
    """
    config_path = kwargs.get("config_path")
    ip = kwargs.get("ip")
    port = kwargs.get("port")
    log_directory = kwargs.get("log_directory")
    system_user = kwargs.get("system_user")

    if config_path is None and (ip is None or port is None):
        print(
               "\nPipeline configuration failed. \nPlease provides configuration file path "
                "or server http ip address & port information."
        )
        return

    if config_path is None:
        config_path = default_config

    with Path(config_path).open("r") as fin:
        config = yaml.safe_load(fin)

    if ip:
        config["ip"] = ip
    if port:
        config["port"] = port
    if log_directory:
        config["log_directory"] = Path(log_directory).resolve().__str__()

    if system_user:
        system_user = system_user.lower()
        if system_user not in ["guest", "host", "arbiter"]:
            raise ValueError(f"system_user {system_user} is not valid. Must be one of (guest, host, arbiter)")
        config["system_setting"] = {"role": system_user}

    with default_config.open("w") as fout:
        yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)

    print("Pipeline configuration succeeded.")


cli.add_command(_init)


if __name__ == '__main__':
    cli()
