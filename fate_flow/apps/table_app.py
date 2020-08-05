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
from fate_flow.manager.data_manager import query_data_view, delete_table
from fate_flow.manager.table_manager.table_operation import get_table
from fate_flow.utils.api_utils import get_json_result
from fate_flow.settings import stat_logger
from arch.api.utils.dtable_utils import get_table_info
from flask import Flask, request

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/delete', methods=['post'])
def table_delete():
    request_data = request.json
    table_name = request_data.get('table_name')
    namespace = request_data.get('namespace')
    table = get_table(name=table_name, namespace=namespace)
    if table:
        table.destroy()
        data = {'table_name': table_name, 'namespace': namespace}
        try:
            table.close()
        except Exception as e:
            stat_logger.exception(e)
    else:
        return get_json_result(retcode=101, retmsg='no find table')
    return get_json_result(data=data)


@manager.route('/<table_func>', methods=['post'])
def dtable(table_func):
    config = request.json
    if table_func == 'table_info':
        table_name, namespace = get_table_info(config=config, create=config.get('create', False))
        if config.get('create', False):
            table_key_count = 0
            table_partition = None
        else:
            table = get_table(name=table_name, namespace=namespace, simple=True)
            if table:
                table_key_count = table.count()
                table_partition = table.get_partitions()
                try:
                    table.close()
                except Exception as e:
                    stat_logger.exception(e)
            else:
                table_key_count = 0
                table_partition = None
        return get_json_result(data={'table_name': table_name, 'namespace': namespace, 'count': table_key_count, 'partition': table_partition})
    else:
        return get_json_result()


