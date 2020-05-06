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
import signal
import sys
import time
from concurrent import futures
import traceback

import grpc
from flask import Flask
from grpc._cython import cygrpc
from werkzeug.serving import run_simple
from werkzeug.wsgi import DispatcherMiddleware

from fate_flow.utils.proto_compatibility import proxy_pb2_grpc
from fate_flow.apps.data_access_app import manager as data_access_app_manager
from fate_flow.apps.job_app import manager as job_app_manager
from fate_flow.apps.model_app import manager as model_app_manager
from fate_flow.apps.pipeline_app import manager as pipeline_app_manager
from fate_flow.apps.table_app import manager as table_app_manager
from fate_flow.apps.tracking_app import manager as tracking_app_manager
from fate_flow.apps.schedule_app import manager as schedule_app_manager
from fate_flow.apps.permission_app import manager as permission_app_manager
from fate_flow.db.db_models import init_database_tables
from fate_flow.driver import dag_scheduler, job_controller, job_detector
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.constant_config import WorkMode, ProcessRole
from fate_flow.manager import queue_manager
from fate_flow.settings import IP, GRPC_PORT, CLUSTER_STANDALONE_JOB_SERVER_PORT, _ONE_DAY_IN_SECONDS, \
    MAX_CONCURRENT_JOB_RUN, stat_logger, API_VERSION, ZOOKEEPER_HOSTS, USE_CONFIGURATION_CENTER, SERVINGS_ZK_PATH, \
    FATE_FLOW_ZK_PATH, HTTP_PORT, FATE_FLOW_MODEL_TRANSFER_PATH, DETECT_TABLE
from fate_flow.utils import job_utils
from fate_flow.utils import session_utils
from fate_flow.utils.api_utils import get_json_result
from fate_flow.utils.authentication_utils import PrivilegeAuth
from fate_flow.utils.grpc_utils import UnaryServicer
from fate_flow.utils.setting_utils import CenterConfig
from arch.api import session

'''
Initialize the manager
'''

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


if __name__ == '__main__':
    manager.url_map.strict_slashes = False
    app = DispatcherMiddleware(
        manager,
        {
            '/{}/data'.format(API_VERSION): data_access_app_manager,
            '/{}/model'.format(API_VERSION): model_app_manager,
            '/{}/job'.format(API_VERSION): job_app_manager,
            '/{}/table'.format(API_VERSION): table_app_manager,
            '/{}/tracking'.format(API_VERSION): tracking_app_manager,
            '/{}/pipeline'.format(API_VERSION): pipeline_app_manager,
            '/{}/schedule'.format(API_VERSION): schedule_app_manager,
            '/{}/permission'.format(API_VERSION): permission_app_manager
        }
    )
    # init
    signal.signal(signal.SIGTERM, job_utils.cleaning)
    signal.signal(signal.SIGCHLD, job_utils.wait_child_process)
    init_database_tables()
    # init runtime config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--standalone_node', default=False, help="if standalone node mode or not ", action='store_true')
    args = parser.parse_args()
    if args.standalone_node:
        RuntimeConfig.init_config(WORK_MODE=WorkMode.STANDALONE)
        RuntimeConfig.init_config(HTTP_PORT=CLUSTER_STANDALONE_JOB_SERVER_PORT)
    session_utils.init_session_for_flow_server()
    RuntimeConfig.init_env()
    RuntimeConfig.set_process_role(ProcessRole.SERVER)
    queue_manager.init_job_queue()
    job_controller.JobController.init()
    PrivilegeAuth.init()
    CenterConfig.init(ZOOKEEPER_HOSTS, USE_CONFIGURATION_CENTER, FATE_FLOW_ZK_PATH, HTTP_PORT,
                      FATE_FLOW_MODEL_TRANSFER_PATH)
    # start job detector
    job_detector.JobDetector(interval=5 * 1000).start()
    # start scheduler
    scheduler = dag_scheduler.DAGScheduler(queue=RuntimeConfig.JOB_QUEUE, concurrent_num=MAX_CONCURRENT_JOB_RUN)
    scheduler.start()
    # start grpc server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                                  (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

    proxy_pb2_grpc.add_DataTransferServiceServicer_to_server(UnaryServicer(), server)
    server.add_insecure_port("{}:{}".format(IP, GRPC_PORT))
    server.start()
    # start http server
    try:
        run_simple(hostname=IP, port=RuntimeConfig.HTTP_PORT, application=app, threaded=True)
        stat_logger.info("FATE Flow server start Successfully")
    except OSError as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGKILL)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGKILL)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        session_utils.clean_server_used_session()
        server.stop(0)
        sys.exit(0)