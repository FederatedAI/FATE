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
from pathlib import Path

import click
from ruamel import yaml

from flow_client.flow_cli.commands import (
    checkpoint, component, data, job, key, model, privilege, provider, queue,
    resource, server, service, table, tag, task, template, test, tracking,
)
from flow_client.flow_cli.utils.cli_utils import prettify


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(short_help='Fate Flow Client', context_settings=CONTEXT_SETTINGS)
@click.pass_context
def flow_cli(ctx):
    '''
    Fate Flow Client
    '''
    ctx.ensure_object(dict)
    if ctx.invoked_subcommand == 'init':
        return

    with open(os.path.join(os.path.dirname(__file__), 'settings.yaml'), 'r') as fin:
        config = yaml.safe_load(fin)
    if not config.get('api_version'):
        raise ValueError('api_version in config is required')
    ctx.obj['api_version'] = config['api_version']

    is_server_conf_exist = False
    if config.get('server_conf_path'):
        conf_path = Path(config['server_conf_path'])
        is_server_conf_exist = conf_path.is_file()

    if is_server_conf_exist:
        server_conf = yaml.safe_load(conf_path.read_text('utf-8'))

        local_conf_path = conf_path.with_name(f'local.{conf_path.name}')
        if local_conf_path.is_file():
            server_conf.update(yaml.safe_load(local_conf_path.read_text('utf-8')))

        ctx.obj['ip'] = server_conf['fateflow']['host']
        ctx.obj['http_port'] = int(server_conf['fateflow']['http_port'])
        ctx.obj['server_url'] = f'http://{ctx.obj["ip"]}:{ctx.obj["http_port"]}/{ctx.obj["api_version"]}'

        http_app_key = None
        http_secret_key = None

        if server_conf.get('authentication', {}).get('client', {}).get('switch'):
            http_app_key = server_conf['authentication']['client']['http_app_key']
            http_secret_key = server_conf['authentication']['client']['http_secret_key']
        else:
            http_app_key = server_conf.get('fateflow', {}).get('http_app_key')
            http_secret_key = server_conf.get('fateflow', {}).get('http_secret_key')

        if http_app_key and http_secret_key:
            ctx.obj['app_key'] = http_app_key
            ctx.obj['secret_key'] = http_secret_key

    elif config.get('ip') and config.get('port'):
        ctx.obj['ip'] = config['ip']
        ctx.obj['http_port'] = int(config['port'])
        ctx.obj['server_url'] = f'http://{ctx.obj["ip"]}:{ctx.obj["http_port"]}/{config["api_version"]}'

        if config.get('app_key') and config.get('secret_key'):
            ctx.obj['app_key'] = config['app_key']
            ctx.obj['secret_key'] = config['secret_key']
    else:
        raise ValueError('Invalid configuration file. Did you run "flow init"?')

    ctx.obj['initialized'] = is_server_conf_exist or (config.get('ip') and config.get('port'))


@flow_cli.command('init', short_help='Flow CLI Init Command')
@click.option('-c', '--server-conf-path', type=click.Path(exists=True),
              help='Server configuration file absolute path.')
@click.option('--ip', type=click.STRING, help='Fate flow server ip address.')
@click.option('--port', type=click.INT, help='Fate flow server port.')
@click.option('--app-key', type=click.STRING, help='APP key for sign requests.')
@click.option('--secret-key', type=click.STRING, help='Secret key for sign requests.')
@click.option('--reset', is_flag=True, default=False,
              help='If specified, initialization settings would be reset to none. Users should init flow again.')
def initialization(**kwargs):
    '''
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
    '''
    with open(os.path.join(os.path.dirname(__file__), 'settings.yaml'), 'r') as fin:
        config = yaml.safe_load(fin)

    if kwargs.get('reset'):
        config['api_version'] = 'v1'
        for i in ('server_conf_path', 'ip', 'port', 'app_key', 'secret_key'):
            config[i] = None

        with open(os.path.join(os.path.dirname(__file__), 'settings.yaml'), 'w') as fout:
            yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)
        prettify(
            {
                'retcode': 0,
                'retmsg': 'Fate Flow CLI has been reset successfully. '
                          'Please do initialization again before you using flow CLI v2.'
            }
        )
    else:
        config['api_version'] = 'v1'
        if kwargs.get('server_conf_path'):
            config['server_conf_path'] = os.path.abspath(kwargs['server_conf_path'])
        for i in ('ip', 'port', 'app_key', 'secret_key'):
            if kwargs.get(i):
                config[i] = kwargs[i]

        if config.get('server_conf_path') or (config.get('ip') and config.get('port')):
            with open(os.path.join(os.path.dirname(__file__), 'settings.yaml'), 'w') as fout:
                yaml.dump(config, fout, Dumper=yaml.RoundTripDumper)
            prettify(
                {
                    'retcode': 0,
                    'retmsg': 'Fate Flow CLI has been initialized successfully.'
                }
            )
        else:
            prettify(
                {
                    'retcode': 100,
                    'retmsg': 'Fate Flow CLI initialization failed. Please provides server configuration file path '
                              'or server http ip address and port information.'
                }
            )


flow_cli.add_command(server.server)
flow_cli.add_command(service.service)
flow_cli.add_command(provider.provider)
flow_cli.add_command(tracking.tracking)
flow_cli.add_command(component.component)
flow_cli.add_command(data.data)
flow_cli.add_command(job.job)
flow_cli.add_command(model.model)
flow_cli.add_command(resource.resource)
flow_cli.add_command(privilege.privilege)
flow_cli.add_command(queue.queue)
flow_cli.add_command(task.task)
flow_cli.add_command(table.table)
flow_cli.add_command(tag.tag)
flow_cli.add_command(checkpoint.checkpoint)
flow_cli.add_command(test.test)
flow_cli.add_command(template.template)
flow_cli.add_command(key.key)


if __name__ == '__main__':
    flow_cli()
