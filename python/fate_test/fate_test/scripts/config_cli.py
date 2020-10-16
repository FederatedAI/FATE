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

from pathlib import Path

import click
from fate_test._client import Clients
from fate_test._config import create_config, default_config, parse_config
from fate_test.scripts._options import SharedOptions


@click.group("config", help="fate_test config")
def config_group():
    """
    config fate_test
    """
    pass


@config_group.command(name="new")
def _new():
    """
    create new fate_test config temperate
    """
    create_config(Path("fate_test_config.yaml"))
    click.echo(f"create config file: fate_test_config.yaml")


@config_group.command(name="edit")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def _edit(ctx, **kwargs):
    """
    edit fate_test config file
    """
    ctx.obj.update(**kwargs)
    config = ctx.obj.get("config")
    click.edit(filename=config)


@config_group.command(name="show")
def _show():
    """
    show fate_test default config path
    """
    click.echo(f"default config path is {default_config()}")


@config_group.command(name="check")
@SharedOptions.get_shared_options(hidden=True)
@click.pass_context
def _config(ctx, **kwargs):
    """
    check connection
    """
    ctx.obj.update(**kwargs)
    config_inst = parse_config(ctx.obj.get("config"))
    with Clients(config_inst) as clients:
        roles = clients.all_roles()
        for r in roles:
            try:
                version, address = clients[r].check_connection()
            except Exception as e:
                click.echo(f"[X]connection fail, role is {r}, exception is {e.args}")
            else:
                click.echo(f"[✓]connection {address} ok, fate version is {version}, role is {r}")
