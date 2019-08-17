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
from arch.task_manager.utils.api_utils import get_json_result
from arch.task_manager.settings import logger
from arch.api.utils.dtable_utils import get_table_info
from flask import Flask, request
from arch.api import storage

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    logger.exception(e)
    return get_json_result(status=100, msg=str(e))


@manager.route('/<table_func>', methods=['post'])
def dtable(table_func):
    config = request.json
    if table_func == 'tableInfo':
        table_name, namespace = get_table_info(config=config, create=config.get('create', False))
        if config.get('create', False):
            table_key_count = 0
        else:
            dtable = storage.get_data_table(name=table_name, namespace=namespace)
            if dtable:
                table_key_count = dtable.count()
            else:
                table_key_count = 0
        return get_json_result(data={'table_name': table_name, 'namespace': namespace, 'count': table_key_count})
    else:
        return get_json_result()
