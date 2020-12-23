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

import operator
import time

from fate_arch.common.base_utils import current_timestamp
from fate_flow.db.db_models import DB, Job, Task
from fate_flow.entity.types import JobStatus, TaskStatus, EndStatus
from fate_arch.common.log import schedule_logger, sql_logger
import peewee


class JobSaver(object):
    STATUS_FIELDS = ["status", "party_status"]

    @classmethod
    def create_job(cls, job_info):
        return cls.create_job_family_entity(Job, job_info)

    @classmethod
    def create_task(cls, task_info):
        return cls.create_job_family_entity(Task, task_info)

    @classmethod
    @DB.connection_context()
    def delete_job(cls, job_id):
        Job.delete().where(Job.f_job_id == job_id)

    @classmethod
    def update_job_status(cls, job_info):
        schedule_logger(job_id=job_info["job_id"]).info("try to update job {} status to {}".format(job_info["job_id"], job_info.get("status")))
        update_status = cls.update_status(Job, job_info)
        if update_status:
            schedule_logger(job_id=job_info["job_id"]).info("update job {} status successfully".format(job_info["job_id"]))
            if EndStatus.contains(job_info.get("status")):
                new_job_info = {}
                for k in ["job_id", "role", "party_id"]:
                    new_job_info[k] = job_info[k]
                new_job_info["tag"] = "job_end"
                cls.update_entity_table(Job, new_job_info)
        else:
            schedule_logger(job_id=job_info["job_id"]).info("update job {} status does not take effect".format(job_info["job_id"]))
        return update_status

    @classmethod
    def update_job(cls, job_info):
        schedule_logger(job_id=job_info["job_id"]).info("try to update job {}".format(job_info["job_id"]))
        update_status = cls.update_entity_table(Job, job_info)
        if update_status:
            schedule_logger(job_id=job_info.get("job_id")).info(f"job {job_info['job_id']} update successfully: {job_info}")
        else:
            schedule_logger(job_id=job_info.get("job_id")).warning(f"job {job_info['job_id']} update does not take effect: {job_info}")
        return update_status

    @classmethod
    def update_task_status(cls, task_info):
        schedule_logger(job_id=task_info["job_id"]).info("try to update job {} task {} {} status".format(task_info["job_id"], task_info["task_id"], task_info["task_version"]))
        update_status = cls.update_status(Task, task_info)
        if update_status:
            schedule_logger(job_id=task_info["job_id"]).info("update job {} task {} {} status successfully: {}".format(task_info["job_id"], task_info["task_id"], task_info["task_version"], task_info))
        else:
            schedule_logger(job_id=task_info["job_id"]).info("update job {} task {} {} status update does not take effect: {}".format(task_info["job_id"], task_info["task_id"], task_info["task_version"], task_info))
        return update_status

    @classmethod
    def update_task(cls, task_info):
        schedule_logger(job_id=task_info["job_id"]).info("try to update job {} task {} {}".format(task_info["job_id"], task_info["task_id"], task_info["task_version"]))
        update_status = cls.update_entity_table(Task, task_info)
        if update_status:
            schedule_logger(job_id=task_info["job_id"]).info("job {} task {} {} update successfully".format(task_info["job_id"], task_info["task_id"], task_info["task_version"]))
        else:
            schedule_logger(job_id=task_info["job_id"]).warning("job {} task {} {} update does not take effect".format(task_info["job_id"], task_info["task_id"], task_info["task_version"]))
        return update_status

    @classmethod
    @DB.connection_context()
    def create_job_family_entity(cls, entity_model, entity_info):
        obj = entity_model()
        obj.f_create_time = current_timestamp()
        for k, v in entity_info.items():
            attr_name = 'f_%s' % k
            if hasattr(entity_model, attr_name):
                setattr(obj, attr_name, v)
        try:
            rows = obj.save(force_insert=True)
            if rows != 1:
                raise Exception("Create {} failed".format(entity_model))
            return obj
        except peewee.IntegrityError as e:
            if e.args[0] == 1062:
                sql_logger(job_id=entity_info.get("job_id", "fate_flow")).warning(e)
            else:
                raise Exception("Create {} failed:\n{}".format(entity_model, e))
        except Exception as e:
            raise Exception("Create {} failed:\n{}".format(entity_model, e))

    @classmethod
    @DB.connection_context()
    def update_status(cls, entity_model, entity_info):
        query_filters = []
        primary_keys = entity_model.get_primary_keys_name()
        for p_k in primary_keys:
            query_filters.append(operator.attrgetter(p_k)(entity_model) == entity_info[p_k.lstrip("f").lstrip("_")])
        objs = entity_model.select().where(*query_filters)
        if objs:
            obj = objs[0]
        else:
            raise Exception("can not found the obj to update")
        update_filters = query_filters[:]
        update_info = {"job_id": entity_info["job_id"]}
        for status_field in cls.STATUS_FIELDS:
            if entity_info.get(status_field) and hasattr(entity_model, f"f_{status_field}"):
                if status_field in ["status", "party_status"]:
                    update_info[status_field] = entity_info[status_field]
                    old_status = getattr(obj, f"f_{status_field}")
                    new_status = update_info[status_field]
                    if_pass = False
                    if isinstance(obj, Task):
                        if TaskStatus.StateTransitionRule.if_pass(src_status=old_status, dest_status=new_status):
                            if_pass = True
                    elif isinstance(obj, Job):
                        if JobStatus.StateTransitionRule.if_pass(src_status=old_status, dest_status=new_status):
                            if_pass = True
                        if EndStatus.contains(new_status) and new_status not in {JobStatus.SUCCESS, JobStatus.CANCELED}:
                            update_filters.append(Job.f_rerun_signal == False)
                    if if_pass:
                        update_filters.append(operator.attrgetter(f"f_{status_field}")(type(obj)) == old_status)
                    else:
                        # not allow update status
                        update_info.pop(status_field)
        return cls.execute_update(old_obj=obj, model=entity_model, update_info=update_info, update_filters=update_filters)

    @classmethod
    @DB.connection_context()
    def update_entity_table(cls, entity_model, entity_info):
        query_filters = []
        primary_keys = entity_model.get_primary_keys_name()
        for p_k in primary_keys:
            query_filters.append(operator.attrgetter(p_k)(entity_model) == entity_info[p_k.lstrip("f").lstrip("_")])
        objs = entity_model.select().where(*query_filters)
        if objs:
            obj = objs[0]
        else:
            raise Exception("can not found the {}".format(entity_model.__class__.__name__))
        update_filters = query_filters[:]
        update_info = {}
        update_info.update(entity_info)
        for _ in cls.STATUS_FIELDS:
            # not allow update status fields by this function
            update_info.pop(_, None)
        if update_info.get("tag") == "job_end" and hasattr(entity_model, "f_tag"):
            if obj.f_start_time:
                update_info["end_time"] = current_timestamp()
                update_info['elapsed'] = update_info['end_time'] - obj.f_start_time
        if update_info.get("progress") and hasattr(entity_model, "f_progress") and update_info["progress"] > 0:
            update_filters.append(operator.attrgetter("f_progress")(entity_model) <= update_info["progress"])
        return cls.execute_update(old_obj=obj, model=entity_model, update_info=update_info, update_filters=update_filters)

    @classmethod
    def execute_update(cls, old_obj, model, update_info, update_filters):
        update_fields = {}
        for k, v in update_info.items():
            attr_name = 'f_%s' % k
            if hasattr(model, attr_name) and attr_name not in model.get_primary_keys_name():
                update_fields[operator.attrgetter(attr_name)(model)] = v
        if update_fields:
            if update_filters:
                operate = old_obj.update(update_fields).where(*update_filters)
            else:
                operate = old_obj.update(update_fields)
            sql_logger(job_id=update_info.get("job_id", "fate_flow")).info(operate)
            return operate.execute() > 0
        else:
            return False

    @classmethod
    @DB.connection_context()
    def query_job(cls, reverse=None, order_by=None, **kwargs):
        filters = []
        for f_n, f_v in kwargs.items():
            attr_name = 'f_%s' % f_n
            if attr_name in ['f_start_time', 'f_end_time', 'f_elapsed']:
                if isinstance(f_v, list):
                    if attr_name == 'f_elapsed':
                        b_timestamp = f_v[0]
                        e_timestamp = f_v[1]
                    else:
                        # time type: %Y-%m-%d %H:%M:%S
                        b_timestamp = str_to_time_stamp(f_v[0])
                        e_timestamp = str_to_time_stamp(f_v[1])
                    filters.append(getattr(Job, attr_name).between(b_timestamp, e_timestamp))
                else:
                    raise Exception('{} need is a list'.format(f_n))
            if hasattr(Job, attr_name):
                filters.append(operator.attrgetter('f_%s' % f_n)(Job) == f_v)
        if filters:
            jobs = Job.select().where(*filters)
            if reverse is not None:
                if not order_by or not hasattr(Job, f"f_{order_by}"):
                    order_by = "create_time"
                if reverse is True:
                    jobs = jobs.order_by(getattr(Job, f"f_{order_by}").desc())
                elif reverse is False:
                    jobs = jobs.order_by(getattr(Job, f"f_{order_by}").asc())
            return [job for job in jobs]
        else:
            return []

    @classmethod
    @DB.connection_context()
    def get_tasks_asc(cls, job_id, role, party_id):
        tasks = Task.select().where(Task.f_job_id == job_id, Task.f_role == role, Task.f_party_id == party_id).order_by(Task.f_create_time.asc())
        tasks_group = cls.get_latest_tasks(tasks=tasks)
        return tasks_group

    @classmethod
    @DB.connection_context()
    def query_task(cls, only_latest=True, reverse=None, order_by=None, **kwargs):
        filters = []
        for f_n, f_v in kwargs.items():
            attr_name = 'f_%s' % f_n
            if hasattr(Task, attr_name):
                filters.append(operator.attrgetter('f_%s' % f_n)(Task) == f_v)
        if filters:
            tasks = Task.select().where(*filters)
        else:
            tasks = Task.select()
        if reverse is not None:
            if not order_by or not hasattr(Task, f"f_{order_by}"):
                order_by = "create_time"
            if reverse is True:
                tasks = tasks.order_by(getattr(Task, f"f_{order_by}").desc())
            elif reverse is False:
                tasks = tasks.order_by(getattr(Task, f"f_{order_by}").asc())
        if only_latest:
            tasks_group = cls.get_latest_tasks(tasks=tasks)
            return list(tasks_group.values())
        else:
            return [task for task in tasks]

    @classmethod
    def get_latest_tasks(cls, tasks):
        tasks_group = {}
        for task in tasks:
            task_key = cls.task_key(task_id=task.f_task_id, role=task.f_role, party_id=task.f_party_id)
            if task_key not in tasks_group:
                tasks_group[task_key] = task
            elif task.f_task_version > tasks_group[task_key].f_task_version:
                # update new version task
                tasks_group[task_key] = task
        return tasks_group

    @classmethod
    def task_key(cls, task_id, role, party_id):
        return f"{task_id}_{role}_{party_id}"


def str_to_time_stamp(time_str):
    time_array = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    time_stamp = int(time.mktime(time_array) * 1000)
    return time_stamp
