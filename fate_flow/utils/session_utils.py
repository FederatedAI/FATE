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

from arch.api import session
from arch.api.utils.log_utils import schedule_logger


class SessionStop(object):
    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('-j', '--job_id', required=True, type=str, help="job id")
        parser.add_argument('-w', '--work_mode', required=True, type=str, help="work mode")
        parser.add_argument('-b', '--backend', required=True, type=str, help="backend")
        args = parser.parse_args()
        job_id = args.job_id
        work_mode = int(args.work_mode)
        backend = int(args.backend)
        session.init(job_id=job_id, mode=work_mode, backend=backend, set_log_dir=False)
        try:
            schedule_logger(job_id.split('_')[0]).info('start stop session {}'.format(session.get_session_id()))
            session.stop()
            schedule_logger(job_id.split('_')[0]).info('stop session {} success'.format(session.get_session_id()))
        except Exception as e:
            pass


if __name__ == '__main__':
    SessionStop.run()
