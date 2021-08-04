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
import traceback

import grpc
from grpc._cython import cygrpc
from werkzeug.serving import run_simple

from fate_flow.utils.proto_compatibility import proxy_pb2_grpc
from fate_flow.apps import app
from fate_flow.db.db_models import init_database_tables as init_flow_db
from fate_arch.storage.metastore.db_models import init_database_tables as init_arch_db
from fate_flow.scheduler.detector import Detector
from fate_flow.scheduler.dag_scheduler import DAGScheduler
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.types import ProcessRole
from fate_flow.manager.resource_manager import ResourceManager
from fate_flow.settings import IP, HTTP_PORT, GRPC_PORT, _ONE_DAY_IN_SECONDS, stat_logger, GRPC_SERVER_MAX_WORKERS, detect_logger
from fate_flow.utils.authentication_utils import PrivilegeAuth
from fate_flow.utils.grpc_utils import UnaryService
from fate_flow.db.db_services import service_db
from fate_flow.utils.xthread import ThreadPoolExecutor
from fate_flow.utils import job_utils
from fate_arch.common.log import schedule_logger


if __name__ == '__main__':
    # init
    # signal.signal(signal.SIGTERM, job_utils.cleaning)
    signal.signal(signal.SIGCHLD, job_utils.wait_child_process)
    # init db
    init_flow_db()
    init_arch_db()
    # init runtime config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--standalone_node', default=False, help="if standalone node mode or not ", action='store_true')
    args = parser.parse_args()
    RuntimeConfig.init_env()
    RuntimeConfig.set_process_role(ProcessRole.DRIVER)
    PrivilegeAuth.init()
    service_db().register_models()
    ResourceManager.initialize()
    Detector(interval=5 * 1000, logger=detect_logger).start()
    DAGScheduler(interval=2 * 1000, logger=schedule_logger()).start()
    thread_pool_executor = ThreadPoolExecutor(max_workers=GRPC_SERVER_MAX_WORKERS)
    stat_logger.info(f"start grpc server thread pool by {thread_pool_executor._max_workers} max workers")
    server = grpc.server(thread_pool=thread_pool_executor,
                         options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                                  (cygrpc.ChannelArgKey.max_receive_message_length, -1)])

    proxy_pb2_grpc.add_DataTransferServiceServicer_to_server(UnaryService(), server)
    server.add_insecure_port("{}:{}".format(IP, GRPC_PORT))
    server.start()
    stat_logger.info("FATE Flow grpc server start successfully")
    # start http server
    try:
        stat_logger.info("FATE Flow http server start...")
        run_simple(hostname=IP, port=HTTP_PORT, application=app, threaded=True)
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
        server.stop(0)
        sys.exit(0)
