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
import subprocess
import re
import sys
from importlib import import_module
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CUSTOM_MODULES = set(['arch', 'eggroll', 'federatedml', 'fate_flow'])
USE_SOURCE_MODULES = set(['antlr4', 'mocks', 'TestTokenStreamRewriter'])


class DummyConfig(object):
    def __init__(self, intersphinx_mapping=None, intersphinx_cache_limit=5, intersphinx_timeout=None):
        self.intersphinx_mapping = intersphinx_mapping or {}
        self.intersphinx_cache_limit = intersphinx_cache_limit
        self.intersphinx_timeout = intersphinx_timeout
        self.tls_verify = True


class DummyApp(object):
    def __init__(self):
        self.config = DummyConfig()


def get_python_standard_modules(version=None):
    version = '{}.{}'.format(sys.version_info[0], sys.version_info[1]) if not version else version
    module_cache_file = 'python{}_modules.csv'.format(version.replace('.', '_'))
    if os.path.exists(module_cache_file):
        print('read python {} standard modules'.format(version))
        modules = list()
        with open(module_cache_file, 'r') as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                modules.append(line.strip())
    else:
        from sphinx.ext.intersphinx import fetch_inventory
        print('fetch python {} standard modules'.format(version))
        url = "http://docs.python.org/{}/objects.inv".format(version)

        modules = sorted(
            list(
                fetch_inventory(DummyApp(), "", url).get("py:module").keys()
            )
        )
        with open(module_cache_file, 'w') as fw:
            fw.write('\n'.join(modules))
    return modules


def search_require_modules(project_dir):
    grep_cmd = "find {} -name '*.py' | grep -v -E '*_pb2\.py' | grep -v -E '*_pb2_grpc\.py' | grep -v -E 'workflow\.py' | xargs -n1 cat | grep -E '^import|^from'".format(project_dir)
    print(grep_cmd)
    p = subprocess.Popen(grep_cmd,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()
    import_lines = stdout.decode('utf-8').strip().split('\n')
    python_standard_modules = get_python_standard_modules('3.6')
    require_modules = set()
    require_lines = dict()
    all_imports = set()
    for line in import_lines:
        import_module = re.sub('^import |^from ', '', line).split(' ')[0].strip()
        require_module = import_module.split('.')[0]
        if len(require_module) == 0:
            continue
        if ',' in require_module:
            tmp = require_module.split(',')
        else:
            tmp = [require_module]
        for r_m in tmp:
            if r_m.startswith('.'):
                continue
            if r_m.endswith('_pb2'):
                continue
            if r_m in USE_SOURCE_MODULES:
                continue
            all_imports.add(line.strip())
            if r_m in python_standard_modules:
                continue
            if r_m in CUSTOM_MODULES:
                continue
            require_modules.add(r_m)
            require_lines[r_m] = line.strip()
    return require_modules, require_lines, all_imports


def conda_env_install(module):
    print('try install: {}'.format(module))
    install_cmd = 'conda install -y {}'.format(module)
    p = subprocess.Popen(install_cmd,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print('try install again: {}'.format(module))
        install_cmd = 'conda install -c conda-forge -y {}'.format(module)
        p = subprocess.Popen(install_cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate()
    return p.returncode


def pip_env_install(module):
    print('try install: {}'.format(module))
    install_cmd = 'pip install {}'.format(module)
    p = subprocess.Popen(install_cmd,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()
    return p.returncode


def try_import(module):
    try:
        import_module(module)
        return 0
    except Exception as e:
        st = pip_env_install(module)
        if st == 0:
            return 1
        else:
            return 2


def check_require(require_modules, require_lines):
    for require_module in require_modules:
        st = try_import(require_module)
        if st == 0:
            continue
        elif st == 1:
            print('installed {}: {}\n'.format(require_module, require_lines[require_module]))
        elif st == 2:
            print('failed installed {}: {}\n'.format(require_module, require_lines[require_module]))


def check_import(all_imports):
    dependent_modules = set()
    dependent_lines = dict()
    for import_code in all_imports:
        python_cmd = "python -c '{}'".format(import_code)
        p = subprocess.Popen(python_cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            # import error
            stdout = stdout.decode('utf-8').strip().split('\n')
            for line in stdout:
                if line.startswith('ModuleNotFoundError:'):
                    require_module = line.strip().split(' ')[-1].strip("'").split('.')[0]
                    print('{}: {}'.format(require_module, import_code))
                    if require_module in CUSTOM_MODULES:
                        pass
                        # code error
                    else:
                        dependent_modules.add(require_module)
                        dependent_lines[require_module] = import_code
    return dependent_modules, dependent_lines


if __name__ == '__main__':
    print('project dir is: {}'.format(PROJECT_DIR))
    print('start search import')
    require_modules, require_lines, all_imports = search_require_modules(PROJECT_DIR)
    print()
    print('has {} require modules'.format(len(require_modules)))
    print(require_modules)
    print()
    check_require(require_modules=require_modules, require_lines=require_lines)
    print()
    dependent_modules, dependent_lines = check_import(all_imports=all_imports)
    print()
    require_modules.update(dependent_modules)
    require_lines.update(dependent_lines)
    check_require(require_modules=require_modules, require_lines=require_lines)
    print()

