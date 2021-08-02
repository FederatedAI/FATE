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
from importlib.util import spec_from_file_location, module_from_spec

from flask import Flask, Blueprint

from fate_flow.settings import API_VERSION, stat_logger
from fate_flow.utils.api_utils import server_error_response


__all__ = ['app']

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
    page.manager = Blueprint(page_name, module_name)
    sys.modules[module_name] = page
    spec.loader.exec_module(page)

    api_version = getattr(page, 'api_version', API_VERSION)
    page_name = getattr(page, 'page_name', page_name)
    app.register_blueprint(page.manager, url_prefix=f'/{api_version}/{page_name}')

stat_logger.info('imported pages: %s', ' '.join(str(path) for path in pages_path))
