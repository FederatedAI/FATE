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
import operator

from fate_arch.common import base_utils
from fate_flow.db.db_models import DB, DBQueue
from fate_flow.entity.types import JobStatus


class JobQueue(object):
    @classmethod
    @DB.connection_context()
    def create_event(cls, job_id, initiator_role, initiator_party_id):
        events = cls.query_event(job_id=job_id, initiator_role=initiator_role,
                                 initiator_party_id=initiator_party_id)
        if events:
            raise RuntimeError(f"job {job_id} {initiator_role} {initiator_party_id} already exists")
        event = DBQueue()
        event.f_job_id = job_id
        event.f_initiator_role = initiator_role
        event.f_initiator_party_id = initiator_party_id
        event.f_job_status = JobStatus.WAITING
        event.f_create_time = base_utils.current_timestamp()
        event.save(force_insert=True)

    @classmethod
    def get_event(cls, job_status):
        events = cls.query_event(job_status=job_status)
        return events

    @classmethod
    @DB.connection_context()
    def update_event(cls, job_id, initiator_role, initiator_party_id, job_status, ttl=None):
        events = cls.query_event(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
        if not events:
            raise RuntimeError(f"job {job_id} {initiator_role} {initiator_party_id} not in queue")
        event = events[0]
        if JobStatus.StateTransitionRule.if_pass(event.f_job_status, job_status):
            if ttl:
                operate = DBQueue.update({DBQueue.f_job_status: job_status}).where(
                    (DBQueue.f_job_id == job_id) &
                    (DBQueue.f_initiator_role == initiator_role) &
                    (DBQueue.f_initiator_party_id == initiator_party_id) &
                    ((DBQueue.f_job_status == event.f_job_status) |
                     (
                             (DBQueue.f_job_status == JobStatus.READY) & (DBQueue.f_update_time < (base_utils.current_timestamp() - ttl))
                     )
                     )
                )
            else:
                operate = DBQueue.update({DBQueue.f_job_status: job_status}).where(
                    (DBQueue.f_job_id == job_id) &
                    (DBQueue.f_initiator_role == initiator_role) &
                    (DBQueue.f_initiator_party_id == initiator_party_id) &
                    (DBQueue.f_job_status == event.f_job_status)
                )
            return operate.execute() > 0
        else:
            raise RuntimeError(f"can not update job status {event.f_job_status} to {job_status} on queue")

    @classmethod
    @DB.connection_context()
    def query_event(cls, **kwargs):
        query_filters = []
        for k, v in kwargs.items():
            attr_name = 'f_%s' % k
            if hasattr(DBQueue, attr_name):
                query_filters.append(operator.attrgetter(attr_name)(DBQueue) == v)
        events = DBQueue.select().where(*query_filters)
        return [event for event in events]

    @classmethod
    @DB.connection_context()
    def delete_event(cls, job_id, initiator_role, initiator_party_id, job_status=None):
        if job_status:
            operate = DBQueue.delete().where(DBQueue.f_job_id == job_id, DBQueue.f_initiator_role == initiator_role,
                                             DBQueue.f_initiator_party_id == initiator_party_id, DBQueue.f_job_status==job_status)
        else:
            operate = DBQueue.delete().where(DBQueue.f_job_id == job_id, DBQueue.f_initiator_role == initiator_role,
                                             DBQueue.f_initiator_party_id == initiator_party_id)
        return operate.execute() > 0

    @DB.connection_context()
    def qsize(self, job_status=None):
        if job_status:
            events = DBQueue.select(DBQueue.f_job_id).where(DBQueue.f_job_status == job_status)
        else:
            events = DBQueue.select(DBQueue.f_job_id)
        return len(events)
