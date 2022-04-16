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
from flow_client.flow_cli.utils.cli_utils import preprocess, access_server


@click.group(short_help="Table Operations")
@click.pass_context
def table(ctx):
    """
    \b
    Provides numbers of table operational commands, including info and delete.
    For more details, please check out the help text.
    """
    pass


@table.command("info", short_help="Query Table Command")
@cli_args.NAMESPACE_REQUIRED
@cli_args.TABLE_NAME_REQUIRED
@click.pass_context
def info(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Query Table Information.

    \b
    - USAGE:
        flow table info -n $NAMESPACE -t $TABLE_NAME
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'table/table_info', config_data)


@table.command("delete", short_help="Delete Table Command")
@cli_args.NAMESPACE
@cli_args.TABLE_NAME
@cli_args.JOBID
@cli_args.ROLE
@cli_args.PARTYID
@cli_args.COMPONENT_NAME
@click.pass_context
def delete(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Delete A Specified Table.

    \b
    - USAGE:
        flow table delete -n $NAMESPACE -t $TABLE_NAME
        flow table delete -j $JOB_ID -r guest -p 9999
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'table/delete', config_data)


@table.command("disable", short_help="Disable Table Command")
@cli_args.NAMESPACE
@cli_args.TABLE_NAME
@click.pass_context
def disable(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Disable A Specified Table.

    \b
    - USAGE:
        flow table disable -n $NAMESPACE -t $TABLE_NAME
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'table/disable', config_data)


@table.command("enable", short_help="Disable Table Command")
@cli_args.NAMESPACE
@cli_args.TABLE_NAME
@click.pass_context
def disable(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Enable A Specified Table.

    \b
    - USAGE:
        flow table enable -n $NAMESPACE -t $TABLE_NAME
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'table/enable', config_data)


@table.command("disable-delete", short_help="Delete Disable Table Command")
@click.pass_context
def disable(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Delete Disable A Specified Table.

    \b
    - USAGE:
        flow table disable-delete
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'table/disable/delete', config_data)


@table.command("add", short_help="Add Table Command")
@cli_args.CONF_PATH
@click.pass_context
def add(ctx, **kwargs):
    """
    - DESCRIPTION:

    \b
    Add a address to fate address.
    Used to be 'table_add'.

    \b
    - USAGE:
        flow table add -c fate_flow/examples/bind_hdfs_table.json
    """
    config_data, _ = preprocess(**kwargs)
    access_server('post', ctx, 'table/add', config_data)


@table.command("bind", short_help="Bind Table Command")
@cli_args.CONF_PATH
@click.option('--drop', is_flag=True, default=False,
              help="If specified, data of old version would be replaced by the current version. "
                   "Otherwise, current upload task would be rejected. (default: False)")
@click.pass_context
def bind(ctx, **kwargs):
    """
    - DESCRIPTION:

    \b
    Bind a address to fate address.
    Used to be 'table_bind'.

    \b
    - USAGE:
        flow table bind -c fate_flow/examples/bind_hdfs_table.json
    """
    config_data, _ = preprocess(**kwargs)
    access_server('post', ctx, 'table/bind', config_data)


@table.command("connector-create", short_help="create or update connector")
@cli_args.CONF_PATH
@click.pass_context
def connector_create_or_update(ctx, **kwargs):
    """
    - DESCRIPTION:

    \b
    Create a connector to fate address.

    \b
    - USAGE:
        flow table connector-create -c fateflow/examples/connector/create_or_update.json
    """
    config_data, _ = preprocess(**kwargs)
    access_server('post', ctx, 'table/connector/create', config_data)


@table.command("connector-query", short_help="query connector info")
@cli_args.CONNECTOR_NAME
@click.pass_context
def connector_query(ctx, **kwargs):
    """
    - DESCRIPTION:

    \b
    query connector info.

    \b
    - USAGE:
        flow table connector-query --connector-name xxx
    """
    config_data, _ = preprocess(**kwargs)
    access_server('post', ctx, 'table/connector/query', config_data)


@table.command("tracking-source", short_help="Tracking Source Table")
@cli_args.NAMESPACE
@cli_args.TABLE_NAME
@click.pass_context
def tracking_source(ctx, **kwargs):
    """
    - DESCRIPTION:

    \b
    tracking a table's parent table

    \b
    - USAGE:
        flow table tracking_source -n $NAMESPACE -t $TABLE_NAME
    """
    config_data, _ = preprocess(**kwargs)
    access_server('post', ctx, 'table/tracking/source', config_data)


@table.command("tracking-job", short_help="Tracking Using Table Job")
@cli_args.NAMESPACE
@cli_args.TABLE_NAME
@click.pass_context
def tracking_job(ctx, **kwargs):
    """
    - DESCRIPTION:

    \b
    tracking jobs of using table

    \b
    - USAGE:
        flow table tracking_job -n $NAMESPACE -t $TABLE_NAME
    """
    config_data, _ = preprocess(**kwargs)
    access_server('post', ctx, 'table/tracking/job', config_data)
