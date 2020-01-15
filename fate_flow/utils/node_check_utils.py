import functools

import requests
from flask import request

from fate_flow.settings import CHECK_NODES_IDENTITY, MANAGER_HOST, MANAGER_PORT


def get_dest_info(request_path, func_name):
    dest_role = request_path.split('/')[2] if 'task' not in func_name else request_path.split('/')[4]
    dest_party_id = request_path.split('/')[3] if 'task' not in func_name else request_path.split('/')[5]
    return dest_role, dest_party_id


def check_nodes(func):
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if CHECK_NODES_IDENTITY:
            dest_role, dest_party_id = get_dest_info(request.path, func.__name__)
            body = {
                'partyId': dest_party_id,
                'role': dest_role,
                'appKey': request.json.get('appKey'),
                'appSecret': request.json.get('appSecret')
            }
            response = requests.post(url="http://{}:{}/node/management/check".format(MANAGER_HOST, MANAGER_PORT), json=body).json()
            if response['code'] != 0:
                raise Exception('Authentication failure: {}'.format(str(response['msg'])))
        return func(*args, **kwargs)
    return _wrapper
