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

from flow_client.flow_cli.utils import cli_args
from flow_client.flow_cli.utils.cli_utils import (preprocess, access_server, check_abs_path)


@click.group(short_help="Component Provider Operations")
@click.pass_context
def provider(ctx):
    """
    \b
    Provides numbers of component operational commands, including metrics, parameters and etc.
    For more details, please check out the help text.
    """
    pass


@provider.command("list", short_help="List All Providers Command")
@click.pass_context
@click.option("-n", "--provider-name", type=click.STRING, help="Provider Name")
def list_providers(ctx, **kwargs):
    config_data, dsl_data = preprocess(**kwargs)
    if kwargs.get("provider_name"):
        access_server("post", ctx, f"provider/{kwargs['provider_name']}/get", config_data)
    else:
        access_server("post", ctx, "provider/get", config_data)


@provider.command("register", short_help="Register New Provider Command")
@cli_args.CONF_PATH
@click.pass_context
def register(ctx, **kwargs):
    config_data, dsl_data = preprocess(**kwargs)
    for p in {"path"}:
        config_data[p] = check_abs_path(config_data.get(p))
    access_server("post", ctx, "provider/register", config_data)


@provider.command("list-components", short_help="List All Components Command")
@click.pass_context
def list_components(ctx, **kwargs):
    config_data, dsl_data = preprocess(**kwargs)
    access_server("post", ctx, "component/get", config_data)
