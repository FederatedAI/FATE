import os
import json
import time
import typing
import tarfile


def start_cluster_standalone_job_server():
    print('use service.sh to start standalone node server....')
    os.system('sh service.sh start --standalone_node')
    time.sleep(5)


def get_project_base_directory():
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir))
    return PROJECT_BASE


def download_from_request(http_response, tar_file_name, extract_dir):
    with open(tar_file_name, 'wb') as fw:
        for chunk in http_response.iter_content(1024):
            if chunk:
                fw.write(chunk)
    tar = tarfile.open(tar_file_name, "r:gz")
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, extract_dir)
    tar.close()
    os.remove(tar_file_name)


def check_config(config: typing.Dict, required_arguments: typing.List):
    no_arguments = []
    error_arguments = []
    for require_argument in required_arguments:
        if isinstance(require_argument, tuple):
            config_value = config.get(require_argument[0], None)
            if isinstance(require_argument[1], (tuple, list)):
                if config_value not in require_argument[1]:
                    error_arguments.append(require_argument)
            elif config_value != require_argument[1]:
                error_arguments.append(require_argument)
        elif require_argument not in config:
            no_arguments.append(require_argument)
    if no_arguments or error_arguments:
        raise Exception('the following arguments are required: {} {}'.format(','.join(no_arguments), ','.join(['{}={}'.format(a[0], a[1]) for a in error_arguments])))


def preprocess(**kwargs):
    config_data = {}
    if 'self' in kwargs:
        kwargs.pop('self')

    if kwargs.get('conf_path'):
        conf_path = os.path.abspath(kwargs.get('conf_path'))
        with open(conf_path, 'r') as conf_fp:
            config_data = json.load(conf_fp)

        if config_data.get('output_path'):
            config_data['output_path'] = os.path.abspath(config_data['output_path'])

        if ('party_id' in kwargs.keys()) or ('role' in kwargs.keys()):
            config_data['local'] = config_data.get('local', {})
            if kwargs.get('party_id'):
                config_data['local']['party_id'] = kwargs.get('party_id')
            if kwargs.get('role'):
                config_data['local']['role'] = kwargs.get('role')

    config_data.update(dict((k, v) for k, v in kwargs.items() if v is not None))

    for key in ['job_id', 'party_id']:
        if isinstance(config_data.get(key), int):
            config_data[key] = str(config_data[key])

    # TODO what if job type is 'predict'
    dsl_data = {}
    if kwargs.get('dsl_path'):
        dsl_path = os.path.abspath(kwargs.get('dsl_path'))
        with open(dsl_path, 'r') as dsl_fp:
            dsl_data = json.load(dsl_fp)
    return config_data, dsl_data








