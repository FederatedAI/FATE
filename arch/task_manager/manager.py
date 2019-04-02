from grpc._cython import cygrpc
import requests
import json
from arch.api import eggroll
from arch.api.proto import proxy_pb2, proxy_pb2_grpc
from arch.api.proto import basic_meta_pb2
from arch.api.utils.log_utils import LoggerFactory
from arch.api.utils import file_utils
from arch.api.utils.parameter_utils import ParameterOverride
from arch.task_manager.adapter.offline_feature.get_feature import GetFeature
from arch.task_manager.job_manager import save_job_info, query_job_info, update_job_info, push_into_job_queue, pop_from_job_queue, \
    get_job_from_queue, running_job_amount, update_job_queue
from arch.task_manager.utils.job_utils import generate_job_id, get_job_directory
from arch.task_manager.utils import cron
from flask.logging import default_handler
from flask import Flask, request, jsonify
import grpc, time, sys
from concurrent import futures
import datetime
import os
import subprocess
import glob
import psutil
from psutil import NoSuchProcess
import uuid
from arch.task_manager.settings import ROLE, SERVERS, IP, GRPC_PORT, HTTP_PORT, LOCAL_URL, PROXY_HOST, PROXY_PORT, \
    PARTY_ID, WORK_MODE, HEADERS, _ONE_DAY_IN_SECONDS, server_conf, MAX_CONCURRENT_JOB_RUN
from arch.task_manager.utils import publish_model

'''
Initialize the manager
'''

manager = Flask(__name__)

'''
Url Configs
'''

JOB_URL = '/job'
WORKFLOW_URL = '/workflow/<job_id>/<module>/<role>'


def get_url(_suffix):
    return "{}{}".format(LOCAL_URL, _suffix)


def get_proxy_data_channel():
    channel = grpc.insecure_channel('{}:{}'.format(PROXY_HOST, PROXY_PORT))
    stub = proxy_pb2_grpc.DataTransferServiceStub(channel)
    return channel, stub


def get_json_result(status=0, msg='success'):
    return jsonify({"status": status, "msg": msg})


class JobCron(cron.Cron):
    def run_do(self):
        if running_job_amount() < MAX_CONCURRENT_JOB_RUN:
            wait_jobs = get_job_from_queue(status="waiting", limit=1)
            if wait_jobs:
                update_job_queue(job_id=wait_jobs[0].get("job_id"), update_data={"status": "ready"})
                self.run_job(wait_jobs[0].get("job_id"), json.loads(wait_jobs[0].get("config")))

    def run_job(self, job_id, config):
        default_runtime_dict = file_utils.load_json_conf('workflow/conf/default_runtime_conf.json')
        setting_conf = file_utils.load_json_conf('workflow/conf/setting_conf.json')
        _job_dir = get_job_directory(job_id=job_id)
        os.makedirs(_job_dir, exist_ok=True)
        ParameterOverride.override_parameter(default_runtime_dict, setting_conf, config, _job_dir)
        manager.logger.info('job_id {} parameters overrode {}'.format(config, _job_dir))
        channel, stub = get_proxy_data_channel()
        for runtime_conf_path in glob.glob(os.path.join(_job_dir, '**', 'runtime_conf.json'), recursive=True):
            runtime_conf = file_utils.load_json_conf(os.path.abspath(runtime_conf_path))
            _role = runtime_conf['local']['role']
            _party_id = runtime_conf['local']['party_id']
            _method = 'POST'
            _module = runtime_conf['module']
            _url = '/workflow/{}/{}/{}'.format(job_id, _module, _role)
            _packet = wrap_grpc_packet(runtime_conf, _method, _url, _party_id, job_id)
            manager.logger.info(
                'Starting workflow job_id:{} party_id:{} role:{} method:{} url:{}'.format(job_id, _party_id,
                                                                                          _role, _method,
                                                                                          _url))
            try:
                _return = stub.unaryCall(_packet)
                manager.logger.info("Grpc unary response: {}".format(_return))
            except grpc.RpcError as e:
                msg = 'job_id:{} party_id:{} role:{} method:{} url:{} Failed to start workflow'.format(job_id,
                                                                                                       _party_id,
                                                                                                       _role, _method,
                                                                                                       _url)
                manager.logger.exception(msg)
                return get_json_result(-101, 'UnaryCall submit to remote manager failed')


@manager.route('/data/<data_func>', methods=['post'])
def download_data(data_func):
    _data = request.json
    _job_id = generate_job_id()
    manager.logger.info('generated job_id {}, body {}'.format(_job_id, _data))
    _job_dir = get_job_directory(_job_id)
    os.makedirs(_job_dir, exist_ok=True)
    _download_module = os.path.join(file_utils.get_project_base_directory(), "arch/api/utils/download.py")
    _upload_module = os.path.join(file_utils.get_project_base_directory(), "arch/api/utils/upload.py")

    if data_func == "download":
        _module = _download_module
    else:
        _module = _upload_module

    try:
        if data_func == "download":
            progs = ["python3",
                     _module,
                     "-j", _job_id,
                     "-c", os.path.abspath(_data.get("config_path"))
                     ]
        else:
            progs = ["python3",
                     _module,
                     "-c", os.path.abspath(_data.get("config_path"))
                     ]

        manager.logger.info('Starting progs: {}'.format(progs))

        std_log = open(os.path.join(_job_dir, 'std.log'), 'w')
        task_pid_path = os.path.join(_job_dir, 'pids')
    
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        else:
            startupinfo = None
        p = subprocess.Popen(progs,
                             stdout=std_log,
                             stderr=std_log,
                             startupinfo=startupinfo
                             )
    
        os.makedirs(task_pid_path, exist_ok=True)
        with open(os.path.join(task_pid_path, data_func + ".pid"), 'w') as f:
            f.truncate()
            f.write(str(p.pid) + "\n")
            f.flush()

        return get_json_result(0, "success, job_id {}".format(_job_id)) 
    except:
        return get_json_result(-104, "failed, job_id {}".format(_job_id)) 


@manager.route('/job/<job_id>', methods=['DELETE'])
def stop_job(job_id):
    _job_dir = get_job_directory(job_id)
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
            manager.logger.info("Grpc unary response: {}".format(_return))
        except grpc.RpcError as e:
            msg = 'job_id:{} party_id:{} role:{} method:{} url:{} Failed to start workflow'.format(job_id,
                                                                                                   _party_id,
                                                                                                   _role, _method,
                                                                                                   _url)
            manager.logger.exception(msg)
            return get_json_result(-101, 'UnaryCall stop to remote manager failed')
    return get_json_result()


@manager.route(JOB_URL, methods=['POST'])
def submit_job():
    _data = request.json
    _job_id = generate_job_id()
    manager.logger.info('generated job_id {}, body {}'.format(_job_id, _data))
    push_into_job_queue(job_id=_job_id, config=_data)
    return get_json_result(0, "success, job_id {}".format(_job_id))


@manager.route('/v1/job/jobStatus/<job_id>', methods=['POST'])
def update_job(job_id):
    request_data = request.json
    print(request_data)
    update_job_info(job_id=job_id, update_data={"status": request_data.get("status")})
    if request_data.get("status") in ["failed", "deleted"]:
        print("stop")
        stop_job(job_id=job_id)
    if request_data.get("status") in ["failed", "deleted", "success"]:
        pop_from_job_queue(job_id=job_id)
    return get_json_result()


@manager.route(WORKFLOW_URL, methods=['POST'])
def start_workflow(job_id, module, role):
    _data = request.json
    _job_dir = get_job_directory(job_id)
    _party_id = str(_data['local']['party_id'])
    _method = _data['WorkFlowParam']['method']
    conf_path_dir = os.path.join(_job_dir, _method, module, role, _party_id)
    os.makedirs(conf_path_dir, exist_ok=True)
    conf_file_path = os.path.join(conf_path_dir, 'runtime_conf.json')
    with open(conf_file_path, 'w+') as f:
        f.truncate()
        f.write(json.dumps(_data, indent=4))
        f.flush()
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    else:
        startupinfo = None
    task_pid_path = os.path.join(_job_dir, 'pids')
    std_log = open(os.path.join(_job_dir, role + '.std.log'), 'w')

    progs = ["python3",
             os.path.join(file_utils.get_project_base_directory(), _data['CodePath']),
             "-j", job_id,
             "-c", os.path.abspath(conf_file_path)
             ]
    print(" ".join(progs))
    manager.logger.info('Starting progs: {}'.format(" ".join(progs)))

    p = subprocess.Popen(progs,
                         stdout=std_log,
                         stderr=std_log,
                         startupinfo=startupinfo
                         )
    os.makedirs(task_pid_path, exist_ok=True)
    with open(os.path.join(task_pid_path, role + ".pid"), 'w') as f:
        f.truncate()
        f.write(str(p.pid) + "\n")
        f.flush()

    job_data = dict()
    job_data["begin_date"] = datetime.datetime.now()
    job_data["status"] = "ready"
    with open(conf_file_path) as fr:
        config = json.load(fr)
    job_data.update(config)
    save_job_info(job_id=job_id, **job_data)
    update_job_queue(job_id=job_id, update_data={"status": "ready"})
    return get_json_result(msg="success, pid is %s" % p.pid)


@manager.route('/workflow/<job_id>', methods=['DELETE'])
def stop_workflow(job_id):
    _job_dir = get_job_directory(job_id)
    task_pid_path = os.path.join(_job_dir, 'pids')
    if os.path.isdir(task_pid_path):
        for pid_file in os.listdir(task_pid_path):
            try:
                if not pid_file.endswith('.pid'):
                    continue
                with open(os.path.join(task_pid_path, pid_file), 'r') as f:
                    pids = f.read().split('\n')
                    for pid in pids:
                        try:
                            if len(pid) == 0:
                                continue
                            manager.logger.debug("terminating process pid:{} {}".format(pid, pid_file))
                            p = psutil.Process(int(pid))
                            for child in p.children(recursive=True):
                                child.kill()
                            p.kill()
                        except NoSuchProcess:
                            continue
            except Exception as e:
                manager.logger.exception("error")
                continue
        update_job_info(job_id=job_id, update_data={"status": "failed", "set_status": "failed"})
    return get_json_result()


@manager.route('/v1/data/importId', methods=['POST'])
def import_id():
    eggroll.init(job_id=generate_job_id(), mode=WORK_MODE)
    request_data = request.json
    table_name_space = "id_library"
    try:
        id_library_info = eggroll.table("info", table_name_space, partition=10, create_if_missing=True, error_if_exist=False)
        if request_data.request("rangeStart") == 0:
            data_id = generate_job_id()
            id_library_info.put("tmp_data_id", data_id)
        else:
            data_id = id_library_info.request("tmp_data_id")
        data_table = eggroll.table(data_id, table_name_space, partition=50, create_if_missing=True, error_if_exist=False)
        for i in request_data.request("ids", []):
            data_table.put(i, "")
        if request_data.request("rangeEnd") and request_data.request("total") and (request_data.request("total") - request_data.request("rangeEnd") == 1):
            # end
            new_id_count = data_table.count()
            if new_id_count == request_data["total"]:
                id_library_info.put(data_id, json.dumps({"salt": request_data.request("salt"), "saltMethod": request_data.request("saltMethod")}))
                old_data_id = id_library_info.request("use_data_id")
                id_library_info.put("use_data_id", data_id)

                # TODO: destroy DTable, should be use a lock
                old_data_table = eggroll.table(old_data_id, table_name_space, partition=50, create_if_missing=True, error_if_exist=False)
                old_data_table.destroy()
                id_library_info.delete(old_data_id)
            else:
                data_table.destroy()
                return get_json_result(2, "The actual amount of data is not equal to total.")
        return get_json_result()
    except Exception as e:
        manager.logger.exception(e)
        return get_json_result(1, "import error.")


@manager.route('/v1/data/requestOfflineFeature', methods=['POST'])
def request_offline_feature():
    request_data = request.json
    try:
        job_id = uuid.uuid1().hex
        response = GetFeature.request(job_id, request_data)
        if response.get("status", 1) == 0:
            job_data = dict()
            job_data.update(request_data)
            job_data["begin_date"] = datetime.datetime.now()
            job_data["status"] = "running"
            job_data["config"] = json.dumps(request_data)
            save_job_info(job_id=job_id, **job_data)
            return get_json_result()
        else:
            return get_json_result(status=1, msg="request offline feature error: %s" % response.get("msg", ""))
    except Exception as e:
        manager.logger.exception(e)
        return get_json_result(status=1, msg="request offline feature error: %s" % e)


@manager.route('/v1/data/importOfflineFeature', methods=['POST'])
def import_offline_feature():
    eggroll.init(job_id=generate_job_id(), mode=WORK_MODE)
    request_data = request.json
    try:
        if not request_data.get("jobId"):
            return get_json_result(status=2, msg="no job id")
        job_id = request_data.get("jobId")
        job_data = query_job_info(job_id=job_id)
        if not job_data:
            return get_json_result(status=3, msg="can not found this job id: %s" % request_data.get("jobId", ""))
        response = GetFeature.import_data(request_data, json.loads(job_data[0]["config"]))
        if response.get("status", 1) == 0:
            update_job_info(job_id=job_id, update_data={"status": "success", "end_date": datetime.datetime.now()})
            return get_json_result()
        else:
            return get_json_result(status=1, msg="request offline feature error: %s" % response.get("msg", ""))
    except Exception as e:
        manager.logger.exception(e)
        return get_json_result(status=1, msg="request offline feature error: %s" % e)


@manager.route('/v1/publish/loadmodel', methods=['POST'])
def load_model():
    config = file_utils.load_json_conf(request.json.get("config_path"))
    _job_id = generate_job_id()
    channel, stub = get_proxy_data_channel()
    for _party_id in config.get("party_ids"):
        config['my_party_id'] = _party_id
        _method = 'POST'
        _url = '/v1/publish/doloadmodel'
        _packet = wrap_grpc_packet(config, _method, _url, _party_id, _job_id)
        manager.logger.info(
            'Starting load model job_id:{} party_id:{} method:{} url:{}'.format(_job_id, _party_id,_method, _url))
        try:
            _return = stub.unaryCall(_packet)
            manager.logger.info("Grpc unary response: {}".format(_return))
        except grpc.RpcError as e:
            msg = 'job_id:{} party_id:{} method:{} url:{} Failed to start load model'.format(_job_id,
                                                                                             _party_id,
                                                                                             _method,
                                                                                             _url)
            manager.logger.exception(msg)
            return get_json_result(-101, 'UnaryCall submit to remote manager failed')


@manager.route('/v1/publish/doloadmodel', methods=['POST'])
def do_load_model():
    request_data = request.json
    try:
        request_data["servings"] = server_conf.get("servers", {}).get("servings", [])
        publish_model.load_model(config_data=request_data)
        return get_json_result()
    except Exception as e:
        manager.logger.exception(e)
        return get_json_result(status=1, msg="load model error: %s" % e)


@manager.route('/v1/publish/publishmodelonline', methods=['POST'])
def publish_model_online():
    request_data = request.json
    try:
        config = file_utils.load_json_conf(request_data.get("config_path"))
        publish_model.publish_online(config_data=config)
        return get_json_result()
    except Exception as e:
        manager.logger.exception(e)
        return get_json_result(status=1, msg="load model error: %s" % e)



def wrap_grpc_packet(_json_body, _method, _url, _dst_party_id=None, job_id=None):
    _src_end_point = basic_meta_pb2.Endpoint(ip=IP, port=GRPC_PORT)
    _src = proxy_pb2.Topic(name=job_id, partyId="{}".format(PARTY_ID), role=ROLE, callback=_src_end_point)
    _dst = proxy_pb2.Topic(name=job_id, partyId="{}".format(_dst_party_id), role=ROLE, callback=None)
    _task = proxy_pb2.Task(taskId=job_id)
    _command = proxy_pb2.Command(name=ROLE)
    _meta = proxy_pb2.Metadata(src=_src, dst=_dst, task=_task, command=_command, operator=_method)
    _data = proxy_pb2.Data(key=_url, value=bytes(json.dumps(_json_body), 'utf-8'))
    return proxy_pb2.Packet(header=_meta, body=_data)


class UnaryServicer(proxy_pb2_grpc.DataTransferServiceServicer):

    def unaryCall(self, _request, context):
        packet = _request
        manager.logger.info("<=== RPC receive: {}".format(packet))
        header = packet.header
        _suffix = packet.body.key
        param_bytes = packet.body.value
        param = bytes.decode(param_bytes)
        job_id = header.task.taskId
        dst = header.src
        method = header.operator
        manager.logger.info("From : {} \tjob_id : {}\tmethod : {}\tparams: {}".format(dst, job_id, method, param))

        action = getattr(requests, method.lower(), None)
        if action:
            resp = action(url=get_url(_suffix), data=param, headers=HEADERS)
        else:
            manager.logger.error("method {} not supported".format(method))

        resp_json = resp.json()

        manager.logger.info("Local http response: {}".format(resp_json))

        return wrap_grpc_packet(resp_json, method, _suffix, dst.partyId, job_id)




if __name__ == '__main__':
    manager.url_map.strict_slashes = False
    LoggerFactory.setDirectory()
    manager.logger.removeHandler(default_handler)
    import logging

    manager.logger.setLevel(logging.DEBUG)
    manager.logger.addHandler(LoggerFactory.get_hanlder('manager'))
    print(manager.logger.handlers)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5),
                         options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                                  (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

    proxy_pb2_grpc.add_DataTransferServiceServicer_to_server(UnaryServicer(), server)
    server.add_insecure_port("{}:{}".format(IP, GRPC_PORT))
    server.start()
    JobCron(interval=10*1000).start()
    manager.run(host=IP, port=HTTP_PORT)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        sys.exit(0)
