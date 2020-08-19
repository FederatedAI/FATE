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
from fate_arch.common.base_utils import current_timestamp
from fate_arch.common import base_utils
from fate_flow.db.db_models import DB, Job, Task
from fate_flow.entity.constant import StatusSet, JobStatus, TaskStatus, EndStatus
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_arch.common.log import schedule_logger, sql_logger
import peewee


class JobSaver(object):
    @classmethod
    def create_job(cls, job_info):
        cls.create_job_family_entity(Job, job_info)

    @classmethod
    def create_task(cls, task_info):
        cls.create_job_family_entity(Task, task_info)

    @classmethod
    def update_job_status(cls, job):
        job_info = {
            "job_id": job.f_job_id,
            "role": job.f_role,
            "party_id": job.f_party_id,
            "status": job.f_status,
        }
        return cls.update_job(job_info=job_info)

    @classmethod
    def update_job(cls, job_info):
        schedule_logger(job_id=job_info["job_id"]).info("Update job {}".format(job_info["job_id"]))
        job_info['run_ip'] = RuntimeConfig.JOB_SERVER_HOST
        if EndStatus.contains(job_info.get("status")):
            job_info['tag'] = 'job_end'
            job_info["end_time"] = current_timestamp()
        return cls.update_job_family_entity(Job, job_info)

    @classmethod
    def update_task_status(cls, task):
        task_info = {
            "task_id": task.f_task_id,
            "task_version": task.f_task_version,
            "role": task.f_role,
            "party_id": task.f_party_id,
            "status": task.f_status
        }
        return cls.update_task(task_info=task_info)

    @classmethod
    def update_task(cls, task_info):
        schedule_logger(job_id=task_info["job_id"]).info("Update job {} task {} {}".format(task_info["job_id"], task_info["task_id"], task_info["task_version"]))
        return cls.update_job_family_entity(Task, task_info)

    @classmethod
    def create_job_family_entity(cls, entity_model, entity_info):
        with DB.connection_context():
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
            except peewee.IntegrityError as e:
                if e.args[0] == 1062:
                    sql_logger(job_id=entity_info.get("job_id", "fate_flow")).warning(e)
                else:
                    raise Exception("Create {} failed:\n{}".format(entity_model, e))
            except Exception as e:
                raise Exception("Create {} failed:\n{}".format(entity_model, e))

    @classmethod
    def gen_update_status_filter(cls, obj, update_info, field, update_filters):
        if_pass = False
        if isinstance(obj, Task):
            if TaskStatus.StateTransitionRule.if_pass(src_status=getattr(obj, f"f_{field}"), dest_status=update_info[field]):
                if_pass = True
        elif isinstance(obj, Job):
            if JobStatus.StateTransitionRule.if_pass(src_status=getattr(obj, f"f_{field}"), dest_status=update_info[field]):
                if_pass = True
        if if_pass:
            update_filters.append(operator.attrgetter(f"f_{field}")(type(obj)) == getattr(obj, f"f_{field}"))
        else:
            # not allow update status
            update_info.pop(field)
        return if_pass

    @classmethod
    def update_job_family_entity(cls, entity_model, entity_info):
        with DB.connection_context():
            query_filters = []
            primary_keys = entity_model._meta.primary_key.field_names
            for p_k in primary_keys:
                query_filters.append(operator.attrgetter(p_k)(entity_model) == entity_info[p_k.lstrip("f_")])
            objs = entity_model.select().where(*query_filters)
            if objs:
                obj = objs[0]
            else:
                raise Exception("Can not found the {}".format(entity_model.__class__.__name__))
            update_filters = query_filters[:]
            if 'status' in entity_info and hasattr(entity_model, "f_status"):
                if entity_model == Job and EndStatus.contains(entity_info["status"]):
                    entity_info['end_time'] = current_timestamp()
                    if obj.f_start_time:
                        entity_info['elapsed'] = entity_info['end_time'] - obj.f_start_time
                cls.gen_update_status_filter(obj=obj, update_info=entity_info, field="status", update_filters=update_filters)
            if "party_status" in entity_info and hasattr(entity_model, "f_party_status"):
                if EndStatus.contains(entity_info["party_status"]):
                    entity_info['end_time'] = current_timestamp()
                    if obj.f_start_time:
                        entity_info['elapsed'] = entity_info['end_time'] - obj.f_start_time
                cls.gen_update_status_filter(obj=obj, update_info=entity_info, field="party_status", update_filters=update_filters)
            if "progress" in entity_info and hasattr(entity_model, "f_progress"):
                update_filters.append(operator.attrgetter("f_progress")(entity_model) <= entity_info["progress"])
            update_fields = {}
            for k, v in entity_info.items():
                attr_name = 'f_%s' % k
                if hasattr(entity_model, attr_name) and attr_name not in primary_keys:
                    update_fields[operator.attrgetter(attr_name)(entity_model)] = v
            if update_fields:
                if update_filters:
                    operate = obj.update(update_fields).where(*update_filters)
                else:
                    operate = obj.update(update_fields)
                sql_logger(job_id=entity_info.get("job_id", "fate_flow")).info(operate)
                return operate.execute() > 0
            else:
                return False

    @classmethod
    def get_job_configuration(cls, job_id, role, party_id, tasks=None):
        with DB.connection_context():
            if tasks:
                jobs_run_conf = {}
                for task in tasks:
                    jobs = Job.select(Job.f_job_id, Job.f_runtime_conf, Job.f_description).where(Job.f_job_id == task.f_job_id)
                    job = jobs[0]
                    jobs_run_conf[job.f_job_id] = job.f_runtime_conf["role_parameters"]["local"]["upload_0"]
                    jobs_run_conf[job.f_job_id]["notes"] = job.f_description
                return jobs_run_conf
            else:
                jobs = Job.select(Job.f_dsl, Job.f_runtime_conf, Job.f_train_runtime_conf).where(Job.f_job_id == job_id,
                                                                                                 Job.f_role == role,
                                                                                                 Job.f_party_id == party_id)
            if jobs:
                job = jobs[0]
                return job.f_dsl, job.f_runtime_conf, job.f_train_runtime_conf
            else:
                return {}, {}, {}

    @classmethod
    def query_job(cls, **kwargs):
        with DB.connection_context():
            filters = []
            for f_n, f_v in kwargs.items():
                attr_name = 'f_%s' % f_n
                if hasattr(Job, attr_name):
                    filters.append(operator.attrgetter('f_%s' % f_n)(Job) == f_v)
            if filters:
                jobs = Job.select().where(*filters)
                return [job for job in jobs]
            else:
                # not allow query all job
                return []

    @classmethod
    def get_tasks_asc(cls, job_id, role, party_id):
        with DB.connection_context():
            tasks = Task.select().where(Task.f_job_id == job_id, Task.f_role == role, Task.f_party_id == party_id).order_by(Task.f_create_time.asc())
            tasks_group = {}
            for task in tasks:
                if task.f_task_id not in tasks_group:
                    tasks_group[task.f_task_id] = task
                elif task.f_task_version > tasks_group[task.f_task_id].f_task_version:
                    # update new version task
                    tasks_group[task.f_task_id] = task
            return tasks_group

    @classmethod
    def query_task(cls, **kwargs):
        with DB.connection_context():
            filters = []
            for f_n, f_v in kwargs.items():
                attr_name = 'f_%s' % f_n
                if hasattr(Task, attr_name):
                    filters.append(operator.attrgetter('f_%s' % f_n)(Task) == f_v)
            if filters:
                tasks = Task.select().where(*filters)
            else:
                tasks = Task.select()
            return [task for task in tasks]

    @classmethod
    def success_task_count(cls, job_id):
        count = 0
        tasks = cls.query_task(job_id=job_id)
        job_component_status = {}
        for task in tasks:
            job_component_status[task.f_component_name] = job_component_status.get(task.f_component_name, set())
            job_component_status[task.f_component_name].add(task.f_status)
        for component_name, role_status in job_component_status.items():
            if len(role_status) == 1 and JobStatus.COMPLETE in role_status:
                count += 1
        return count

    @classmethod
    def update_job_progress(cls, job_id, dag, current_task_id):
        role, party_id = cls.query_job_info(job_id)
        component_count = len(dag.get_dependency(role=role, party_id=int(party_id))['component_list'])
        success_count = cls.success_task_count(job_id=job_id)
        job = Job()
        job.f_progress = float(success_count) / component_count * 100
        job.f_update_time = current_timestamp()
        job.f_current_tasks = base_utils.json_dumps([current_task_id])
        return job
