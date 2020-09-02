import unittest

from fate_flow.db.db_models import BaseDataBase

from fate_flow.entity.constant_config import WorkMode

from fate_flow.settings import WORK_MODE
from fate_flow.utils import job_utils

from fate_flow.manager.queue_manager import MysqlQueue, ListQueue

DB = BaseDataBase().database_connection
if WORK_MODE == WorkMode.CLUSTER:
    job_queue = MysqlQueue()
elif WORK_MODE == WorkMode.STANDALONE:
    job_queue = ListQueue()


class TestQueueUtil(unittest.TestCase):
    def test_queue_put(self):
        job_id = job_utils.generate_job_id()
        event = {
            'job_id': job_id,
            "initiator_role": 'loacl',
            "initiator_party_id": 0
        }
        # queue put
        job_queue.put_event(event)

        # queue qsize
        n = job_queue.qsize()
        if n:
            # queue get
            job_event = job_queue.get()
            self.assertIsNotNone(job_event)


if __name__ == '__main__':
    unittest.main()