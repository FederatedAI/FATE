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
import argparse
import functools

from arch.api import session
from arch.api.utils.log_utils import schedule_logger
from arch.api.utils.core_utils import fate_uuid
from fate_flow.settings import stat_logger, DETECT_TABLE
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.constant_config import ProcessRole


class SessionStop(object):
    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('-j', '--job_id', required=True, type=str, help="job id")
        parser.add_argument('-w', '--work_mode', required=True, type=str, help="work mode")
        parser.add_argument('-b', '--backend', required=True, type=str, help="backend")
        parser.add_argument('-c', '--command', required=True, type=str, help="command")
        args = parser.parse_args()
        session_job_id = args.job_id
        fate_job_id = session_job_id.split('_')[0]
        work_mode = int(args.work_mode)
        backend = int(args.backend)
        command = args.command
        session.init(job_id=session_job_id, mode=work_mode, backend=backend, set_log_dir=False)
        try:
            schedule_logger(fate_job_id).info('start {} session {}'.format(command, session.get_session_id()))
            if command == 'stop':
                session.stop()
            elif command == 'kill':
                session.kill()
            else:
                schedule_logger(fate_job_id).info('{} session {} failed, this command is not supported'.format(command, session.get_session_id()))
            schedule_logger(fate_job_id).info('{} session {} success'.format(command, session.get_session_id()))
        except Exception as e:
            pass


def init_session_for_flow_server():
    # Options are used with different backend on demand
    session.init(job_id="session_used_by_fate_flow_server_{}".format(fate_uuid()),
                 mode=RuntimeConfig.WORK_MODE,
                 backend=RuntimeConfig.BACKEND,
                 options={"eggroll.session.processors.per.node": 1})
    # init session detect table
    detect_table = session.table(namespace=DETECT_TABLE[0], name=DETECT_TABLE[1], partition=DETECT_TABLE[2])
    detect_table.destroy()
    detect_table = session.table(namespace=DETECT_TABLE[0], name=DETECT_TABLE[1], partition=DETECT_TABLE[2])
    detect_table.put_all(enumerate(range(DETECT_TABLE[2])))
    stat_logger.info("init detect table {} {} for session {}".format(detect_table.get_namespace(),
                                                                     detect_table.get_name(),
                                                                     session.get_session_id()))
    stat_logger.info("init session {} for fate flow server successfully".format(session.get_session_id()))


def clean_server_used_session():
    used_session_id = None
    try:
        used_session_id = session.get_session_id()
        session.stop()
    except:
        pass
    session.exit()
    stat_logger.info("clean session {} for fate flow server done".format(used_session_id))


def session_detect():
    def _out_wrapper(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if RuntimeConfig.PROCESS_ROLE in [ProcessRole.SERVER]:
                for i in range(3):
                    try:
                        stat_logger.info("detect session {} by table {} {}".format(
                            session.get_session_id(), DETECT_TABLE[0], DETECT_TABLE[1]))
                        stat_logger.info("start count table {} {}".format(DETECT_TABLE[0], DETECT_TABLE[1]))
                        count = session.table(namespace=DETECT_TABLE[0], name=DETECT_TABLE[1]).count()
                        stat_logger.info("table {} {} count is {}".format(DETECT_TABLE[0], DETECT_TABLE[1], count))
                        if count != DETECT_TABLE[2]:
                            raise Exception("session {} count error".format(session.get_session_id()))
                        stat_logger.info("session {} is ok".format(session.get_session_id()))
                        break
                    except Exception as e:
                        stat_logger.exception(e)
                        stat_logger.info("start init new session")
                        try:
                            clean_server_used_session()
                            init_session_for_flow_server()
                        except Exception as e:
                            stat_logger.exception(e)
                            stat_logger.info("init new session failed.")
                else:
                    stat_logger.error("init new session failed.")
            else:
                # If in executor pass. TODO: detect and restore the session in executor
                pass
            return func(*args, **kwargs)
        return _wrapper
    return _out_wrapper


if __name__ == '__main__':
    SessionStop.run()
