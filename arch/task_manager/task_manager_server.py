from grpc._cython import cygrpc
import json
from arch.api.proto import proxy_pb2_grpc
from arch.api import eggroll
from arch.api.utils import file_utils, dtable_utils
from arch.api.utils.parameter_utils import ParameterOverride
from arch.task_manager.job_manager import pop_from_job_queue, \
    get_job_from_queue, running_job_amount, update_job_queue, get_job_directory, \
    save_job_info, is_job_initiator, query_job_by_id, check_job_process, show_job_queue
from arch.task_manager.utils.api_utils import get_json_result, federated_api, local_api
from arch.task_manager.utils import cron
from flask import Flask, request
import grpc, time, sys
from concurrent import futures
import os
import glob
import copy
from arch.task_manager.settings import IP, GRPC_PORT, HTTP_PORT, _ONE_DAY_IN_SECONDS, MAX_CONCURRENT_JOB_RUN, logger
from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple
from arch.task_manager.utils.grpc_utils import UnaryServicer
from arch.task_manager.settings import PARTY_ID, DEFAULT_WORKFLOW_DATA_TYPE, WORK_MODE
from arch.task_manager.apps.data_access import manager as data_access_manager
from arch.task_manager.apps.machine_learning_model import manager as model_manager
from arch.task_manager.apps.workflow import manager as workflow_manager

'''
Initialize the manager
'''

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    logger.exception(e)
    return get_json_result(status=100, msg=str(e))


class JobCron(cron.Cron):
    def run_do(self):
        logger.info("{} job are running.".format(running_job_amount()))
        try:
            if running_job_amount() < MAX_CONCURRENT_JOB_RUN:
                wait_jobs = get_job_from_queue(status="waiting", limit=1)
                if wait_jobs:
                    wait_job = wait_jobs[0]
                    run_job_id = wait_job.job_id
                    try:
                        run_job_success = self.run_job(job_id=run_job_id, config=json.loads(wait_job.config))
                    except Exception as e:
                        run_job_success = False
                        logger.exception(e)
                    if run_job_success:
                        update_job_queue(job_id=run_job_id,
                                         role=wait_job.role,
                                         party_id=wait_job.party_id,
                                         save_data={"status": "ready"})
                    else:
                        pop_from_job_queue(job_id=run_job_id)
            logger.info("check waiting jobs done.")
            self.check_job()
        except Exception as e:
            logger.exception(e)

    def fill_runtime_conf_table_info(self, runtime_conf, default_runtime_conf):
        if not runtime_conf.get('scene_id') or not runtime_conf.get('gen_table'):
            return
        table_config = copy.deepcopy(runtime_conf)
        workflow_param = runtime_conf.get('WorkFlowParam')
        default_workflow_param = default_runtime_conf.get('WorkFlowParam')
        for data_type in DEFAULT_WORKFLOW_DATA_TYPE:
            name_param = '{}_table'.format(data_type)
            namespace_param = '{}_namespace'.format(data_type)
            table_config['data_type'] = data_type
            input_output = data_type.split('_')[-1]
            if (not workflow_param.get(name_param)
                or workflow_param.get(name_param) == default_workflow_param.get(name_param)) \
                    and (not workflow_param.get(namespace_param)
                         or workflow_param.get(namespace_param) == default_workflow_param.get(namespace_param)):
                table_name, namespace = dtable_utils.get_table_info(config=table_config,
                                                                    create=(False if input_output == 'input' else True))
                workflow_param[name_param] = table_name
                workflow_param[namespace_param] = namespace

    def run_job(self, job_id, config):
        default_runtime_dict = file_utils.load_json_conf('workflow/conf/default_runtime_conf.json')
        setting_conf = file_utils.load_json_conf('workflow/conf/setting_conf.json')
        _job_dir = get_job_directory(job_id=job_id)
        os.makedirs(_job_dir, exist_ok=True)
        ParameterOverride.override_parameter(default_runtime_dict, setting_conf, config, _job_dir)
        logger.info('job_id {} parameters overrode {}'.format(config, _job_dir))
        run_job_success = True
        job_param = dict()
        job_param['job_id'] = job_id
        job_param['initiator'] = PARTY_ID
        for runtime_conf_path in glob.glob(os.path.join(_job_dir, '**', 'runtime_conf.json'), recursive=True):
            runtime_conf = file_utils.load_json_conf(os.path.abspath(runtime_conf_path))
            runtime_conf['JobParam'] = job_param
            _role = runtime_conf['local']['role']
            _party_id = runtime_conf['local']['party_id']
            _module = runtime_conf['module']
            self.fill_runtime_conf_table_info(runtime_conf=runtime_conf,
                                              default_runtime_conf=default_runtime_dict)
            st, msg = federated_api(job_id=job_id,
                                    method='POST',
                                    url='/workflow/{}/{}/{}'.format(job_id, _module, _role),
                                    party_id=_party_id,
                                    json_body=runtime_conf)
            if st == 0:
                save_job_info(job_id=job_id,
                              role=_role,
                              party_id=_party_id,
                              save_info={"status": "ready", "initiator": PARTY_ID})
            else:
                run_job_success = False
        logger.info("run job done")
        return run_job_success

    def check_job(self):
        running_jobs = get_job_from_queue(status="running", limit=0)
        for running_job in running_jobs:
            if not check_job_process(running_job.pid):
                local_api(method='POST',
                          suffix='/job/jobStatus/{}/{}/{}'.format(running_job.job_id,
                                                                  running_job.role,
                                                                  running_job.party_id),
                          json_body={'status': 'failed'}
                          )


@manager.route('/<job_id>', methods=['DELETE'])
def stop_job(job_id):
    _job_dir = get_job_directory(job_id)
    all_party = []
    for runtime_conf_path in glob.glob(os.path.join(_job_dir, '**', 'runtime_conf.json'), recursive=True):
        runtime_conf = file_utils.load_json_conf(os.path.abspath(runtime_conf_path))
        for _role, _party_ids in runtime_conf['role'].items():
            all_party.extend([(_role, _party_id) for _party_id in _party_ids])
    all_party = set(all_party)
    logger.info('start send stop job to {}'.format(','.join([i[0] for i in all_party])))
    _method = 'DELETE'
    for _role, _party_id in all_party:
        federated_api(job_id=job_id,
                      method=_method,
                      url='/workflow/{}/{}/{}'.format(job_id, _role, _party_id),
                      party_id=_party_id)
    return get_json_result(job_id=job_id)


@manager.route('/jobStatus/<job_id>/<role>/<party_id>', methods=['POST'])
def update_job(job_id, role, party_id):
    request_data = request.json
    logger.info('job_id:{} role:{} party_id:{} status:{}'.format(job_id, role, party_id, request_data.get('status')))
    job_info = save_job_info(job_id=job_id,
                             role=role,
                             party_id=party_id,
                             save_info={"status": request_data.get("status")})
    update_job_queue(job_id=job_id,
                     role=role,
                     party_id=party_id,
                     save_data={"status": request_data.get("status")})
    if request_data.get("status") in ["success", "failed", "deleted"]:
        pop_from_job_queue(job_id=job_id)
    if is_job_initiator(job_info.initiator, party_id):
        # I am job initiator
        logger.info('i am job {} initiator'.format(job_id))
        # check job status
        jobs = query_job_by_id(job_id=job_id)
        job_status = set([job.status for job in jobs])
        do_stop_job = False
        if 'failed' in job_status or 'deleted' in job_status:
            do_stop_job = True
        elif len(job_status) == 1 and 'success' in job_status:
            do_stop_job = True
        if do_stop_job:
            stop_job(job_id=job_id)
    else:
        # send job status to initiator
        if not request_data.get('initiatorUpdate', False):
            request_data['initiatorUpdate'] = True
            federated_api(job_id=job_id,
                          method='POST',
                          url='/job/jobStatus/{}/{}/{}'.format(job_id, role, party_id),
                          party_id=job_info.initiator,
                          json_body=request_data)
    return get_json_result(job_id=job_id)


@manager.route('/jobStatus/<job_id>', methods=['POST'])
def query_job(job_id):
    jobs = query_job_by_id(job_id=job_id)
    return get_json_result(job_id=job_id, data=[job.to_json() for job in jobs])


@manager.route('/queueStatus', methods=['POST'])
def job_queue_status():
    jobs = show_job_queue()
    return get_json_result(data=dict([(job.job_id, job.to_json()) for job in jobs]))


if __name__ == '__main__':
    eggroll.init(mode=WORK_MODE)
    manager.url_map.strict_slashes = False
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                                  (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

    proxy_pb2_grpc.add_DataTransferServiceServicer_to_server(UnaryServicer(), server)
    server.add_insecure_port("{}:{}".format(IP, GRPC_PORT))
    server.start()
    JobCron(interval=5*1000).start()
    app = DispatcherMiddleware(
        manager,
        {
            '/data': data_access_manager,
            '/model': model_manager,
            '/workflow': workflow_manager,
            '/job': manager
        }
    )
    run_simple(hostname=IP, port=HTTP_PORT, application=app, threaded=True)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        sys.exit(0)
