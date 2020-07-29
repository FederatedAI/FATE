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
from arch.api.utils import file_utils
from arch.api.utils.core_utils import get_lan_ip
from fate_flow.settings import SERVERS, ROLE, API_VERSION
from .flow_cli import (component, data, job, model,
                       privilege, queue, task, table)

server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(short_help="Fate Flow Client", context_settings=CONTEXT_SETTINGS)
@click.pass_context
def flow_cli(ctx):
    """
    Fate Flow Client
    """
    ip = server_conf.get(SERVERS).get(ROLE).get('host')
    if ip in ['localhost', '127.0.0.1']:
        ip = get_lan_ip()
    ctx.ensure_object(dict)
    ctx.obj['http_port'] = server_conf.get(SERVERS).get(ROLE).get('http.port')
    ctx.obj['server_url'] = "http://{}:{}/{}".format(ip, ctx.obj['http_port'], API_VERSION)


flow_cli.add_command(component.component)
flow_cli.add_command(data.data)
flow_cli.add_command(job.job)
flow_cli.add_command(model.model)
flow_cli.add_command(privilege.privilege)
flow_cli.add_command(queue.queue)
flow_cli.add_command(task.task)
flow_cli.add_command(table.table)
