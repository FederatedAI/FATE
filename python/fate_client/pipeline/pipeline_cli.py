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

from flow_sdk.client import FlowClient

default_config = Path(__file__).parent.joinpath("config.yaml").resolve()


@click.group("Pipeline Config Tool")
def cli():
    pass


@click.command(name="init")
@click.option("-c", "--pipeline-conf-path", "config_path", type=click.Path(exists=True),
              help="Path to pipeline configuration file.")
@click.option("-d", "--log-directory", type=click.Path(),
              help="Path to pipeline logs directory.")
@click.option("--ip", type=click.STRING, help="Fate Flow server ip address.")
@click.option("--port", type=click.INT, help="Fate Flow server port.")
@click.option("--app-key", type=click.STRING, help="app key for request to Fate Flow server")
@click.option("--secret-key", type=click.STRING, help="secret key for request to Fate Flow server")
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
    app_key = kwargs.get("app_key")
    secret_key = kwargs.get("secret_key")

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
    if app_key:
        config["app_key"] = app_key
    if secret_key:
        config["secret_key"] = secret_key

    if system_user:
        system_user = system_user.lower()
        if system_user not in ["guest", "host", "arbiter"]:
            raise ValueError(f"system_user {system_user} is not valid. Must be one of (guest, host, arbiter)")
        config["system_setting"] = {"role": system_user}

    with default_config.open("w") as fout:
        yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)

    print("Pipeline configuration succeeded.")


@click.group("config", help="pipeline config tool")
def config_group():
    """
    pipeline config
    """
    pass


@config_group.command(name="show")
def _show():
    """
        \b
        - DESCRIPTION:
            Show pipeline config details for Flow server.

        \b
        - USAGE:
            pipeline config show
    """
    with Path(default_config).open("r") as fin:
        config = yaml.safe_load(fin)
        click.echo(f"\nPipeline Config: {yaml.dump(config)}")


@config_group.command(name="check")
def _check():
    """
        \b
        - DESCRIPTION:
            Check for Flow server status and Flow version.

        \b
        - USAGE:
            pipeline config check
    """
    from pipeline.backend import config as conf
    if conf.FlowConfig.IP is None:
        click.echo(f"Flow server ip not yet configured. Please specify setting with pipeline initialization tool.")
        return
    if conf.FlowConfig.PORT is None:
        click.echo(f"Flow server port not yet configured. Please specify setting with pipeline initialization tool.")

    client = FlowClient(ip=conf.FlowConfig.IP, port=conf.FlowConfig.PORT, version=conf.SERVER_VERSION)
    version = client.remote_version.fate_flow()
    if version is None:
        click.echo(f"Flow server not responsive. Please check flow server ip and port setting.")
    else:
        click.echo(f"Flow server status normal, Flow version: {version}")


cli.add_command(_init)
cli.add_command(config_group)


if __name__ == '__main__':
    cli()
