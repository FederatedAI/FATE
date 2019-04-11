from grpc._cython import cygrpc
import json
from arch.api.proto import proxy_pb2_grpc
from arch.api.utils import file_utils
from arch.api.utils.parameter_utils import ParameterOverride
from arch.task_manager.job_manager import update_job_by_id, push_into_job_queue, pop_from_job_queue, \
    get_job_from_queue, running_job_amount, update_job_queue, generate_job_id, get_job_directory, get_json_result
from arch.task_manager.utils import cron
from flask import Flask, request
import grpc, time, sys
from concurrent import futures
import os
import glob
from arch.task_manager.settings import IP, GRPC_PORT, HTTP_PORT, _ONE_DAY_IN_SECONDS, MAX_CONCURRENT_JOB_RUN, logger
from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple
from arch.task_manager.utils.grpc_utils import wrap_grpc_packet, get_proxy_data_channel, UnaryServicer
from arch.task_manager.apps.data_access import manager as data_access_manager
from arch.task_manager.apps.mlmodel import manager as model_manager
from arch.task_manager.apps.workflow import manager as workflow_manager

'''
Initialize the manager
'''

manager = Flask(__name__)

'''
Url Configs
'''


class JobCron(cron.Cron):
    def run_do(self):
        logger.info("{} job are running.".format(running_job_amount()))
        try:
            if running_job_amount() < MAX_CONCURRENT_JOB_RUN:
                wait_jobs = get_job_from_queue(status="waiting", limit=1)
                if wait_jobs:
                    update_job_queue(job_id=wait_jobs[0].get("job_id"), update_data={"status": "ready"})
                    self.run_job(wait_jobs[0].get("job_id"), json.loads(wait_jobs[0].get("config")))
            logger.info("check waiting jobs done.")
        except Exception as e:
            logger.exception(e)

    def run_job(self, job_id, config):
        default_runtime_dict = file_utils.load_json_conf('workflow/conf/default_runtime_conf.json')
        setting_conf = file_utils.load_json_conf('workflow/conf/setting_conf.json')
        _job_dir = get_job_directory(job_id=job_id)
        os.makedirs(_job_dir, exist_ok=True)
        ParameterOverride.override_parameter(default_runtime_dict, setting_conf, config, _job_dir)
        logger.info('job_id {} parameters overrode {}'.format(config, _job_dir))
        channel, stub = get_proxy_data_channel()
        run_job_success = True
        for runtime_conf_path in glob.glob(os.path.join(_job_dir, '**', 'runtime_conf.json'), recursive=True):
            runtime_conf = file_utils.load_json_conf(os.path.abspath(runtime_conf_path))
            _role = runtime_conf['local']['role']
            _party_id = runtime_conf['local']['party_id']
            _method = 'POST'
            _module = runtime_conf['module']
            _url = '/workflow/{}/{}/{}'.format(job_id, _module, _role)
            _packet = wrap_grpc_packet(runtime_conf, _method, _url, _party_id, job_id)
            logger.info(
                'Starting workflow job_id:{} party_id:{} role:{} method:{} url:{}'.format(job_id, _party_id,
                                                                                          _role, _method,
                                                                                          _url))
            try:
                _return = stub.unaryCall(_packet)
                logger.info("Grpc unary response: {}".format(_return))
                logger.info("{} done".format(runtime_conf_path))
            except grpc.RpcError as e:
                msg = 'job_id:{} party_id:{} role:{} method:{} url:{} Failed to start workflow'.format(job_id,
                                                                                                       _party_id,
                                                                                                       _role, _method,
                                                                                                       _url)
                logger.exception(msg)
                run_job_success = False
            except Exception as e:
                logger.exception(e)
                run_job_success = False
        channel.close()
        if not run_job_success:
            pop_from_job_queue(job_id=job_id)
        logger.info("run job done")


@manager.route('/new', methods=['POST'])
def submit_job():
    _data = request.json
    _job_id = generate_job_id()
    logger.info('generated job_id {}, body {}'.format(_job_id, _data))
    try:
        push_into_job_queue(job_id=_job_id, config=_data)
        return get_json_result(0, "success, job_id {}".format(_job_id))
    except Exception as e:
        return get_json_result(1, "failed, error: {}".format(e))


@manager.route('/<job_id>', methods=['DELETE'])
def stop_job(job_id):
    _job_dir = get_job_directory(job_id)
    try:
        for runtime_conf_path in glob.glob(os.path.join(_job_dir, '**', 'runtime_conf.json'), recursive=True):
            runtime_conf = file_utils.load_json_conf(os.path.abspath(runtime_conf_path))
            _role = runtime_conf['local']['role']
            _party_id = runtime_conf['local']['party_id']
            _url = '/workflow/{}'.format(job_id)
            _method = 'DELETE'
            _packet = wrap_grpc_packet({}, _method, _url, _party_id, job_id)
            channel, stub = get_proxy_data_channel()
            try:
                _return = stub.unaryCall(_packet)
                logger.info("Grpc unary response: {}".format(_return))
            except grpc.RpcError as e:
                msg = 'job_id:{} party_id:{} role:{} method:{} url:{} Failed to start workflow'.format(job_id,
                                                                                                       _party_id,
                                                                                                       _role, _method,
                                                                                                       _url)
                logger.exception(msg)
                return get_json_result(-101, 'UnaryCall stop to remote manager failed')
        return get_json_result()
    except Exception as e:
        logger.exception(e)
        return get_json_result(-102, str(e))


@manager.route('/jobStatus/<job_id>', methods=['POST'])
def update_job(job_id):
    request_data = request.json
    update_job_by_id(job_id=job_id, update_data={"status": request_data.get("status")})
    update_job_queue(job_id=job_id, update_data={"status": request_data.get("status")})
    if request_data.get("status") in ["failed", "deleted"]:
        stop_job(job_id=job_id)
    if request_data.get("status") in ["failed", "deleted", "success"]:
        pop_from_job_queue(job_id=job_id)
    return get_json_result()


if __name__ == '__main__':
    manager.url_map.strict_slashes = False
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                                  (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

    proxy_pb2_grpc.add_DataTransferServiceServicer_to_server(UnaryServicer(), server)
    server.add_insecure_port("{}:{}".format(IP, GRPC_PORT))
    server.start()
    JobCron(interval=5*1000).start()
    app = DispatcherMiddleware(manager,{
        '/data': data_access_manager,
        '/model': model_manager,
        '/workflow': workflow_manager,
        '/job': manager
    })
    run_simple(hostname=IP, port=HTTP_PORT, application=app, threaded=True)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        sys.exit(0)
