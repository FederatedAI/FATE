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


@click.group(short_help="Tag Operations")
@click.pass_context
def tag(ctx):
    """
    \b
    Provides numbers of model tags operational commands.
    For more details, please check out the help text.
    """
    pass


@tag.command("create", short_help="Create Tag Command")
@cli_args.TAG_NAME_REQUIRED
@cli_args.TAG_DESCRIPTION
@click.pass_context
def create_tag(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Create Tag Command.

    \b
    - USAGE:
        flow tag create -t $TAG_NAME -d $TEST_DESCRIPTION
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/tag/create', config_data)


@tag.command("query", short_help="Retrieve Tag Command")
@cli_args.TAG_NAME_REQUIRED
@click.option("--with-model", is_flag=True, default=False,
              help="If specified, the information of models which have the "
                   "tag custom queried would be displayed.")
@click.pass_context
def query_tag(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Retrieve Tag Command.

    \b
    - USAGE:
        flow tag query -t $TAG_NAME
        flow tag query -t $TAG_NAME --with-model
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/tag/retrieve', config_data)


@tag.command("update", short_help="Update Tag Command")
@cli_args.TAG_NAME_REQUIRED
@click.option("--new-tag-name", type=click.STRING, required=False,
              help="New Tag Name.")
@click.option("--new-tag-desc", type=click.STRING, required=False,
              help="New Tag Description. Note that if there are some whitespaces in description, "
                   "please make sure the description text is enclosed in double quotation marks.")
@click.pass_context
def update_tag(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Update Tag Command.

    \b
    - USAGE:
        flow tag update -t tag1 --new-tag-name tag2
        flow tag update -t tag1 --new-tag-desc "This is the new description."
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/tag/update', config_data)


@tag.command("delete", short_help="Delete Tag Command")
@cli_args.TAG_NAME_REQUIRED
@click.pass_context
def delete_tag(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        Delete Tag Command. Notice that the information of model would not be discarded even though the tag is removed.

    \b
    - USAGE:
        flow tag delete -t tag1
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/tag/destroy', config_data)


@tag.command("list", short_help="List Tag Command")
@cli_args.LIMIT
@click.pass_context
def list_tag(ctx, **kwargs):
    """
    \b
    - DESCRIPTION:
        List Tag Command.

    \b
    - USAGE:
        flow tag list
        flow tag list -l 3
    """
    config_data, dsl_data = preprocess(**kwargs)
    access_server('post', ctx, 'model/tag/list', config_data)
