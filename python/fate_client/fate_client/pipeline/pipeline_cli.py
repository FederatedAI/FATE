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

default_config = Path(__file__).parent.joinpath("pipeline_config.yaml").resolve()


@click.group(name="pipeline")
def pipeline_group():
    ...


@pipeline_group.group("init", help="pipeline init")
def init_group():
    """
    pipeline config
    """
    pass


@init_group.command(name="fateflow")
@click.option("--ip", type=click.STRING, help="Fate Flow server ip address.")
@click.option("--port", type=click.INT, help="Fate Flow server port.")
def _init_flow_service(**kwargs):
    ip = kwargs.get("ip")
    port = kwargs.get("port")

    config_path = default_config

    with Path(config_path).open("r") as fin:
        config = yaml.safe_load(fin)

    flow_config = dict()
    if ip:
        flow_config["ip"] = ip
    if port:
        flow_config["port"] = port

    if flow_config:
        config["fate_flow"] = flow_config

    with default_config.open("w") as fout:
        yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)

    print("Pipeline configuration succeeded.")


@init_group.command(name="standalone")
@click.option("--job_dir", type=click.STRING, help="job working directory for standalone pipeline jobs")
def _init_standalone(**kwargs):
    job_dir = kwargs.get("job_dir")

    config_path = default_config
    with Path(config_path).open("r") as fin:
        config = yaml.safe_load(fin)

    if job_dir:
        config["standalone"]["job_dir"] = job_dir

    with default_config.open("w") as fout:
        yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)

    print("Pipeline configuration succeeded.")


@init_group.command(name="config_file")
@click.option("--path", type=click.STRING, help="config pipeline by file directly")
def _config(**kwargs):
    path = kwargs.get("path")
    with Path(path).open("r") as fin:
        config = yaml.safe_load(fin)

    with default_config.open("w") as fout:
        yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)


@pipeline_group.command(name="show")
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


