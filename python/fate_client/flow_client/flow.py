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
import os
import click
from ruamel import yaml
from flow_client.flow_cli.utils.cli_utils import prettify
from flow_client.flow_cli.commands import (component, data, job, model, queue, task, table, tag)


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(short_help="Fate Flow Client", context_settings=CONTEXT_SETTINGS)
@click.pass_context
def flow_cli(ctx):
    """
    Fate Flow Client
    """
    ctx.ensure_object(dict)
    with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "r") as fin:
        config = yaml.safe_load(fin)
    if config.get("server_conf_path"):
        is_server_conf_exist = os.path.exists(config.get("server_conf_path"))
    else:
        is_server_conf_exist = False

    if is_server_conf_exist:
        try:
            with open(config.get("server_conf_path")) as server_conf_fp:
                server_conf = yaml.safe_load(server_conf_fp)
            ip = server_conf.get("fateflow", {}).get("host")
            ctx.obj["http_port"] = server_conf.get("fateflow", {}).get("http_port")
            ctx.obj["server_url"] = "http://{}:{}/{}".format(ip, ctx.obj["http_port"], config.get("api_version"))
        except Exception:
            return
    else:
        if config.get("ip") and config.get("port"):
            ip = config.get("ip")
            ctx.obj["http_port"] = int(config.get("port"))
            ctx.obj["server_url"] = "http://{}:{}/{}".format(ip, ctx.obj["http_port"], config.get("api_version"))

    ctx.obj["init"] = is_server_conf_exist or (config.get("ip") and config.get("port"))


@flow_cli.command("init", short_help="Flow CLI Init Command")
@click.option("-c", "--server-conf-path", type=click.Path(exists=True),
              help="Server configuration file absolute path.")
@click.option("--ip", type=click.STRING, help="Fate flow server ip address.")
@click.option("--port", type=click.INT, help="Fate flow server port.")
@click.option("--reset", is_flag=True, default=False,
              help="If specified, initialization settings would be reset to none. Users should init flow again.")
def initialization(**kwargs):
    """
    \b
    - DESCRIPTION:
        Flow CLI Init Command. Custom can choose to provide an absolute path of server conf file,
        or provide ip address and http port of a valid fate flow server. Notice that, if custom
        provides both, the server conf would be loaded in priority. In this case, ip address and
        http port would be ignored.

    \b
    - USAGE:
        flow init -c /data/projects/fate/python/conf/service_conf.yaml
        flow init --ip 127.0.0.1 --port 9380
    """
    with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "r") as fin:
        config = yaml.safe_load(fin)

    if kwargs.get('reset'):
        config["server_conf_path"] = None
        config["ip"] = None
        config["port"] = None
        with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "w") as fout:
            yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)
        prettify(
            {
                "retcode": 0,
                "retmsg": "Fate Flow CLI has been reset successfully. "
                          "Please do initialization again before you using flow CLI v2."
            }
        )
    else:
        if kwargs.get("server_conf_path"):
            config["server_conf_path"] = os.path.abspath(kwargs.get("server_conf_path"))
        if kwargs.get("ip"):
            config["ip"] = kwargs.get("ip")
        if kwargs.get("port"):
            config["port"] = kwargs.get("port")
        if kwargs.get("server_conf_path") or (kwargs.get("ip") and kwargs.get("port")):
            with open(os.path.join(os.path.dirname(__file__), "settings.yaml"), "w") as fout:
                yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)
            prettify(
                {
                    "retcode": 0,
                    "retmsg": "Fate Flow CLI has been initialized successfully."
                }
            )
        else:
            prettify(
                {
                    "retcode": 100,
                    "retmsg": "Fate Flow CLI initialization failed. Please provides server configuration file path "
                              "or server http ip address and port information."
                }
            )


flow_cli.add_command(component.component)
flow_cli.add_command(data.data)
flow_cli.add_command(job.job)
flow_cli.add_command(model.model)
# flow_cli.add_command(privilege.privilege)
flow_cli.add_command(queue.queue)
flow_cli.add_command(task.task)
flow_cli.add_command(table.table)
flow_cli.add_command(tag.tag)