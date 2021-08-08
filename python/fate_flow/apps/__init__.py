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
import sys
from pathlib import Path
from time import time
from base64 import b64encode
from hmac import HMAC
from logging.config import dictConfig
from datetime import datetime, timezone
from importlib.util import spec_from_file_location, module_from_spec

from flask import Flask, Blueprint, request

from fate_flow.settings import API_VERSION, Settings, stat_logger, MAX_TIMESTAMP_INTERVAL
from fate_flow.utils.api_utils import server_error_response, error_response
from fate_flow.settings import API_VERSION, stat_logger, access_logger
from fate_flow.utils.api_utils import server_error_response
import logging


__all__ = ['app']

logger = logging.getLogger('flask.app')
for h in access_logger.handlers:
    logger.addHandler(h)

# https://stackoverflow.com/questions/56905756/how-to-make-flask-log-to-stdout-instead-of-stderr
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
app.url_map.strict_slashes = False
app.errorhandler(500)(server_error_response)

pages_dir = [
    Path(__file__).parent,
    Path(__file__).parent.parent / 'scheduling_apps'
]
pages_path = [j for i in pages_dir for j in i.glob('*_app.py')]

for path in pages_path:
    page_name = path.stem.rstrip('_app')
    module_name = '.'.join(path.parts[path.parts.index('fate_flow'):-1] + (page_name, ))

    spec = spec_from_file_location(module_name, path)
    page = module_from_spec(spec)
    page.app = app
    page.manager = Blueprint(page_name, module_name)
    sys.modules[module_name] = page
    spec.loader.exec_module(page)

    api_version = getattr(page, 'api_version', API_VERSION)
    page_name = getattr(page, 'page_name', page_name)

    if not isinstance(page.manager, Blueprint):
        raise TypeError('page.manager should be {!r}, got {!r}. filepath: {!s}'.format(Blueprint, page.manager, path))
    app.register_blueprint(page.manager, url_prefix=f'/{api_version}/{page_name}')

stat_logger.info('imported pages: %s', ' '.join(str(path) for path in pages_path))


@app.before_request
def authentication():
    if not (Settings.HTTP_APP_KEY and Settings.HTTP_SECRET_KEY):
        return

    required_headers = {
        'TIMESTAMP',
        'NONCE'
        'APP_KEY',
        'SIGNATURE',
    }
    if required_headers - set(request.headers):
        return error_response(401)

    try:
        timestamp = datetime.fromtimestamp(int(request.headers['TIMESTAMP']) / 1000, tz=timezone.utc)
    except Exception:
        return error_response(400, 'Invalid TIMESTAMP')

    now = time()
    if not now - MAX_TIMESTAMP_INTERVAL < timestamp < now + MAX_TIMESTAMP_INTERVAL:
        return error_response(425, f'TIMESTAMP is more than {MAX_TIMESTAMP_INTERVAL} seconds away from the server time')

    if not request.headers['NONCE']:
        return error_response(400, 'Invalid NONCE')

    if request.headers['APP_KEY'] != Settings.HTTP_APP_KEY:
        return error_response(401, 'Unknown APP_KEY')

    signature = b64encode(HMAC(Settings.HTTP_SECRET_KEY.encode('ascii'), b'\n'.join([
        request.headers['TIMESTAMP'].encode('ascii'),
        request.headers['NONCE'].encode('ascii'),
        request.headers['APP_KEY'].encode('ascii'),
        request.full_path.encode('ascii'),
        request.data,
    ]), 'sha1').digest()).decode('ascii')
    if signature != request.headers['SIGNATURE']:
        return error_response(403)
