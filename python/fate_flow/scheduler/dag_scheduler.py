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
import traceback

from fate_arch.common import FederatedCommunicationType, FederatedMode
from fate_arch.common.base_utils import json_loads, current_timestamp
from fate_arch.common.log import schedule_logger
from fate_arch.common import WorkMode, Backend, EngineType
from fate_arch.common import conf_utils, string_utils
from fate_flow.db.db_models import Job
from fate_flow.scheduler import FederatedScheduler
from fate_flow.scheduler import TaskScheduler
from fate_flow.operation import JobSaver
from fate_flow.entity.types import JobStatus, TaskStatus, EndStatus, StatusSet, SchedulingStatusCode, ResourceOperation, FederatedSchedulingStatusCode, RunParameters
from fate_flow.operation import Tracker
from fate_flow.controller import JobController
from fate_flow.settings import FATE_BOARD_DASHBOARD_ENDPOINT, DEFAULT_TASK_PARALLELISM, DEFAULT_TASK_CORES_PER_NODE, \
    DEFAULT_TASK_MEMORY_PER_NODE, DEFAULT_FEDERATED_STATUS_COLLECT_TYPE
from fate_flow.utils import detect_utils, job_utils, schedule_utils
from fate_flow.utils.service_utils import ServiceUtils
from fate_flow.utils import model_utils
from fate_flow.operation import JobQueue
from fate_flow.utils.cron import Cron
from fate_flow.manager import ResourceManager
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine


class DAGScheduler(Cron):
    @classmethod
    def submit(cls, job_data, job_id=None):
        if not job_id:
            job_id = job_utils.generate_job_id()
        schedule_logger(job_id).info('submit job, job_id {}, body {}'.format(job_id, job_data))
        job_dsl = job_data.get('job_dsl', {})
        job_runtime_conf = job_data.get('job_runtime_conf', {})
        job_initiator = job_runtime_conf['initiator']
        job_parameters = RunParameters(**job_runtime_conf['job_parameters'])
        cls.backend_compatibility(job_parameters=job_parameters)

        job_utils.check_job_runtime_conf(job_runtime_conf)
        if job_parameters.job_type != 'predict':
            # generate job model info
            job_parameters.model_id = model_utils.gen_model_id(job_runtime_conf['role'])
            job_parameters.model_version = job_id
            train_runtime_conf = {}
        else:
            detect_utils.check_config(job_parameters.to_dict(), ['model_id', 'model_version'])
            # get inference dsl from pipeline model as job dsl
            tracker = Tracker(job_id=job_id, role=job_initiator['role'], party_id=job_initiator['party_id'],
                              model_id=job_parameters.model_id, model_version=job_parameters.model_version)
            pipeline_model = tracker.get_output_model('pipeline')
            if not job_dsl:
                job_dsl = json_loads(pipeline_model['Pipeline'].inference_dsl)
            train_runtime_conf = json_loads(pipeline_model['Pipeline'].train_runtime_conf)

        path_dict = job_utils.save_job_conf(job_id=job_id,
                                            job_dsl=job_dsl,
                                            job_runtime_conf=job_runtime_conf,
                                            train_runtime_conf=train_runtime_conf,
                                            pipeline_dsl=None)

        job = Job()
        job.f_job_id = job_id
        job.f_dsl = job_dsl
        job_runtime_conf["job_parameters"] = job_parameters.to_dict()
        job.f_runtime_conf = job_runtime_conf
        job.f_train_runtime_conf = train_runtime_conf
        job.f_roles = job_runtime_conf['role']
        job.f_work_mode = job_parameters.work_mode
        job.f_initiator_role = job_initiator['role']
        job.f_initiator_party_id = job_initiator['party_id']

        initiator_role = job_initiator['role']
        initiator_party_id = job_initiator['party_id']
        if initiator_party_id not in job_runtime_conf['role'][initiator_role]:
            schedule_logger(job_id).info("initiator party id error:{}".format(initiator_party_id))
            raise Exception("initiator party id error {}".format(initiator_party_id))

        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job_dsl,
                                                       runtime_conf=job_runtime_conf,
                                                       train_runtime_conf=train_runtime_conf)

        cls.set_default_job_parameters(job_parameters=job_parameters)

        # update runtime conf
        job_runtime_conf["job_parameters"] = job_parameters.to_dict()
        job.f_runtime_conf = job_runtime_conf

        status_code, response = FederatedScheduler.create_job(job=job)
        if status_code != FederatedSchedulingStatusCode.SUCCESS:
            raise Exception("create job failed: {}".format(response))

        if job_parameters.work_mode == WorkMode.CLUSTER:
            # Save the status information of all participants in the initiator for scheduling
            for role, party_ids in job_runtime_conf["role"].items():
                for party_id in party_ids:
                    if role == job_initiator['role'] and party_id == job_initiator['party_id']:
                        continue
                    JobController.initialize_tasks(job_id, role, party_id, False, job_initiator, job_parameters, dsl_parser)

        # push into queue
        try:
            JobQueue.create_event(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
        except Exception as e:
            raise Exception(f'push job into queue failed:\n{e}')

        schedule_logger(job_id).info(
            'submit job successfully, job id is {}, model id is {}'.format(job.f_job_id, job_parameters.model_id))
        board_url = "http://{}:{}{}".format(
            ServiceUtils.get_item("fateboard", "host"),
            ServiceUtils.get_item("fateboard", "port"),
            FATE_BOARD_DASHBOARD_ENDPOINT).format(job_id, job_initiator['role'], job_initiator['party_id'])
        logs_directory = job_utils.get_job_log_directory(job_id)
        return job_id, path_dict['job_dsl_path'], path_dict['job_runtime_conf_path'], logs_directory, \
               {'model_id': job_parameters.model_id, 'model_version': job_parameters.model_version}, board_url

    @classmethod
    def backend_compatibility(cls, job_parameters: RunParameters):
        # Compatible with previous 1.5 versions
        if job_parameters.computing_engine is None or job_parameters.federation_engine is None:
            if job_parameters.work_mode is None or job_parameters.backend is None:
                raise RuntimeError("unable to find compatible backend engines")
            work_mode = WorkMode(job_parameters.work_mode)
            backend = Backend(job_parameters.backend)
            if backend == Backend.EGGROLL:
                if work_mode == WorkMode.CLUSTER:
                    job_parameters.computing_engine = ComputingEngine.EGGROLL
                    job_parameters.federation_engine = FederationEngine.EGGROLL
                    job_parameters.storage_engine = StorageEngine.EGGROLL
                else:
                    job_parameters.computing_engine = ComputingEngine.STANDALONE
                    job_parameters.federation_engine = FederationEngine.STANDALONE
                    job_parameters.storage_engine = StorageEngine.STANDALONE
            elif backend == Backend.SPARK:
                job_parameters.computing_engine = ComputingEngine.SPARK
                job_parameters.federation_engine = FederationEngine.RABBITMQ
                job_parameters.storage_engine = StorageEngine.HDFS
                # add mq info
                federation_info = {}
                federation_info['union_name'] = string_utils.random_string(4) 
                federation_info['policy_id'] = string_utils.random_string(10)
                job_parameters.federation_info = federation_info
        if job_parameters.federated_mode is None:
            if job_parameters.computing_engine in [ComputingEngine.EGGROLL, ComputingEngine.SPARK]:
                job_parameters.federated_mode = FederatedMode.MULTIPLE
            elif job_parameters.computing_engine in [ComputingEngine.STANDALONE]:
                job_parameters.federated_mode = FederatedMode.SINGLE

    @classmethod
    def set_default_job_parameters(cls, job_parameters: RunParameters):
        if job_parameters.task_parallelism is None:
            job_parameters.task_parallelism = DEFAULT_TASK_PARALLELISM
        computing_engine_info = ResourceManager.get_backend_registration_info(engine_type=EngineType.COMPUTING, engine_name=job_parameters.computing_engine)
        if job_parameters.task_nodes is None:
            job_parameters.task_nodes = computing_engine_info.f_nodes
        if job_parameters.task_cores_per_node is None:
            job_parameters.task_cores_per_node = DEFAULT_TASK_CORES_PER_NODE
        if job_parameters.task_memory_per_node is None:
            job_parameters.task_memory_per_node = DEFAULT_TASK_MEMORY_PER_NODE
        if job_parameters.federated_status_collect_type is None:
            job_parameters.federated_status_collect_type = DEFAULT_FEDERATED_STATUS_COLLECT_TYPE

    def run_do(self):
        schedule_logger().info("start schedule waiting jobs")
        events = JobQueue.get_event(job_status=JobStatus.WAITING)
        schedule_logger().info(f"have {len(events)} waiting jobs")
        if len(events):
            # FIFO
            event = events[0]
            schedule_logger().info(f"schedule waiting job {event.f_job_id}")
            try:
                self.schedule_waiting_jobs(event=event)
            except Exception as e:
                schedule_logger(event.f_job_id).exception(e)
                schedule_logger(event.f_job_id).error(f"schedule waiting job {event.f_job_id} failed")
        schedule_logger().info("schedule waiting jobs finished")

        schedule_logger().info("start schedule running jobs")
        jobs = JobSaver.query_job(is_initiator=True, status=JobStatus.RUNNING)
        schedule_logger().info(f"have {len(jobs)} running jobs")
        for job in jobs:
            try:
                self.schedule_running_job(job=job)
            except Exception as e:
                schedule_logger(job.f_job_id).exception(e)
                schedule_logger(job.f_job_id).error(f"schedule job {job.f_job_id} failed")
        schedule_logger().info("schedule running jobs finished")

        # some ready job exit before start
        schedule_logger().info("start schedule ready jobs")
        events = JobQueue.get_event(job_status=JobStatus.READY)
        schedule_logger().info(f"have {len(events)} ready jobs")
        for event in events:
            try:
                self.schedule_ready_jobs(event=event)
            except Exception as e:
                schedule_logger(event.f_job_id).exception(e)
                schedule_logger(event.f_job_id).error(f"schedule ready job {event.f_job_id} failed:\n{e}")
        schedule_logger().info("schedule ready jobs finished")

        # update canceled job status
        schedule_logger().info("start schedule canceled jobs")
        events = JobQueue.get_event(job_status=JobStatus.CANCELED)
        schedule_logger().info(f"have {len(events)} canceled jobs")
        for event in events:
            try:
                self.schedule_canceled_jobs(event=event)
            except Exception as e:
                schedule_logger(event.f_job_id).exception(e)
                schedule_logger(event.f_job_id).error(f"schedule canceled job {event.f_job_id} failed:\n{e}")
        schedule_logger().info("schedule canceled jobs finished")

    @classmethod
    def schedule_waiting_jobs(cls, event):
        job_id, initiator_role, initiator_party_id, = event.f_job_id, event.f_initiator_role, event.f_initiator_party_id,
        update_status = JobQueue.update_event(job_id=job_id,
                                              initiator_role=initiator_role,
                                              initiator_party_id=initiator_party_id,
                                              job_status=JobStatus.READY)
        if not update_status:
            schedule_logger(job_id).info(f"job {job_id} may be handled by another scheduler")
            return
        # apply resource on all party
        jobs = JobSaver.query_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id)
        job = jobs[0]
        status_code, federated_response = FederatedScheduler.resource_for_job(job=job, operation_type=ResourceOperation.APPLY)
        if status_code == FederatedSchedulingStatusCode.SUCCESS:
            cls.start_job(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
            JobQueue.update_event(job_id=job_id,
                                  initiator_role=initiator_role,
                                  initiator_party_id=initiator_party_id,
                                  job_status=JobStatus.RUNNING)
        else:
            # rollback resource
            rollback_party = {}
            failed_party = {}
            for dest_role in federated_response.keys():
                for dest_party_id in federated_response[dest_role].keys():
                    retcode = federated_response[dest_role][dest_party_id]["retcode"]
                    if retcode == 0:
                        rollback_party[dest_role] = rollback_party.get(dest_role, [])
                        rollback_party[dest_role].append(dest_party_id)
                    else:
                        failed_party[dest_role] = failed_party.get(dest_role, [])
                        failed_party[dest_role].append(dest_party_id)
            schedule_logger(job_id).info("job {} apply resource failed on {}, rollback {}".format(
                job_id,
                ",".join([",".join([f"{_r}:{_p}" for _p in _ps]) for _r, _ps in failed_party.items()]),
                ",".join([",".join([f"{_r}:{_p}" for _p in _ps]) for _r, _ps in rollback_party.items()]),
            ))
            if rollback_party:
                status_code, federated_response = FederatedScheduler.resource_for_job(job=job, operation_type=ResourceOperation.RETURN, specific_dest=rollback_party)
                if status_code != FederatedSchedulingStatusCode.SUCCESS:
                    schedule_logger(job_id).info(f"job {job_id} return resource failed:\n{federated_response}")
            else:
                schedule_logger(job_id).info(f"job {job_id} no party should be rollback resource")
            update_status = JobQueue.update_event(job_id=job_id,
                                                  initiator_role=initiator_role,
                                                  initiator_party_id=initiator_party_id,
                                                  job_status=JobStatus.WAITING)
            schedule_logger(job_id).info(f"update job {job_id} status to waiting {update_status}")

    @classmethod
    def schedule_ready_jobs(cls, event):
        job_id, initiator_role, initiator_party_id, = event.f_job_id, event.f_initiator_role, event.f_initiator_party_id,
        update_status = JobQueue.update_event(job_id=job_id,
                                              initiator_role=initiator_role,
                                              initiator_party_id=initiator_party_id,
                                              job_status=JobStatus.WAITING,
                                              ttl=5*60*1000)
        schedule_logger(job_id).info(f"update job {job_id} ready status to waiting {update_status}")

    @classmethod
    def start_job(cls, job_id, initiator_role, initiator_party_id):
        schedule_logger(job_id=job_id).info("Try to start job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))
        job_info = {}
        job_info["job_id"] = job_id
        job_info["role"] = initiator_role
        job_info["party_id"] = initiator_party_id
        job_info["status"] = JobStatus.RUNNING
        job_info["party_status"] = JobStatus.RUNNING
        job_info["start_time"] = current_timestamp()
        job_info["tag"] = 'end_waiting'
        jobs = JobSaver.query_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id)
        if jobs:
            job = jobs[0]
            FederatedScheduler.start_job(job=job)
            schedule_logger(job_id=job_id).info("start job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))
        else:
            schedule_logger(job_id=job_id).error("can not found job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))

    @classmethod
    def schedule_running_job(cls, job):
        schedule_logger(job_id=job.f_job_id).info("scheduling job {}".format(job.f_job_id))
        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job.f_dsl,
                                                       runtime_conf=job.f_runtime_conf,
                                                       train_runtime_conf=job.f_train_runtime_conf)
        task_scheduling_status_code, tasks = TaskScheduler.schedule(job=job, dsl_parser=dsl_parser)
        tasks_status = [task.f_status for task in tasks]
        new_job_status = cls.calculate_job_status(task_scheduling_status_code=task_scheduling_status_code, tasks_status=tasks_status)
        total, finished_count = cls.calculate_job_progress(tasks_status=tasks_status)
        new_progress = float(finished_count) / total * 100
        schedule_logger(job_id=job.f_job_id).info("Job {} status is {}, calculate by task status list: {}".format(job.f_job_id, new_job_status, tasks_status))
        if new_job_status != job.f_status or new_progress != job.f_progress:
            # Make sure to update separately, because these two fields update with anti-weight logic
            if new_progress != job.f_progress:
                job.f_progress = new_progress
                FederatedScheduler.sync_job(job=job, update_fields=["progress"])
                cls.update_job_on_initiator(initiator_job=job, update_fields=["progress"])
            if new_job_status != job.f_status:
                job.f_status = new_job_status
                if EndStatus.contains(job.f_status):
                    FederatedScheduler.save_pipelined_model(job=job)
                FederatedScheduler.sync_job_status(job=job)
                cls.update_job_on_initiator(initiator_job=job, update_fields=["status"])
        if EndStatus.contains(job.f_status):
            cls.finish(job=job, end_status=job.f_status)
        schedule_logger(job_id=job.f_job_id).info("finish scheduling job {}".format(job.f_job_id))

    @classmethod
    def schedule_canceled_jobs(cls, event):
        job_id, initiator_role, initiator_party_id = event.f_job_id, event.f_initiator_role, event.f_initiator_party_id
        jobs = JobSaver.query_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id, is_initiator=True)
        for job in jobs:
            # update canceled job status
            cls.schedule_running_job(job=job)

    @classmethod
    def align_input_data_partitions(cls, job: Job, dsl_parser):
        job_args = dsl_parser.get_args_input()
        dataset = JobController.get_dataset(is_initiator=True,
                                            role=job.f_initiator_role,
                                            party_id=job.f_initiator_party_id,
                                            roles=job.f_roles, job_args=job_args)
        input_data_aligned_partitions = 0  # Large integers are not used
        status, partys_response = FederatedScheduler.align_args(job=job, command_body=dataset)
        schedule_logger(job_id=job.f_job_id).info("align partition partys response: {}".format(partys_response))
        for role, party_args in partys_response.items():
            for party_id, response in party_args.items():
                if not response["retcode"]:
                    partitions = response.get('data').get('min_input_data_partition')
                    if partitions:
                        if input_data_aligned_partitions == 0 or partitions < input_data_aligned_partitions:
                            input_data_aligned_partitions = partitions
        return input_data_aligned_partitions

    @classmethod
    def rerun_job(cls, job_id, initiator_role, initiator_party_id, component_name):
        schedule_logger(job_id=job_id).info(f"try to rerun job {job_id} on initiator {initiator_role} {initiator_party_id}")
        jobs = JobSaver.query_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id)
        if jobs:
            job = jobs[0]
        else:
            raise RuntimeError(f"can not found job {job_id} on initiator {initiator_role} {initiator_party_id}")
        if component_name != job_utils.job_virtual_component_name():
            tasks = JobSaver.query_task(job_id=job_id, role=initiator_role, party_id=initiator_party_id, component_name=component_name)
        else:
            tasks = JobSaver.query_task(job_id=job_id, role=initiator_role, party_id=initiator_party_id)
        should_rerun = False
        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job.f_dsl,
                                                       runtime_conf=job.f_runtime_conf,
                                                       train_runtime_conf=job.f_train_runtime_conf)
        for task in tasks:
            if task.f_status != TaskStatus.COMPLETE and task.f_party_status != TaskStatus.COMPLETE:
                # stop old version task
                FederatedScheduler.stop_task(job=job, task=task, stop_status=task.f_status)
                FederatedScheduler.clean_task(job=job, task=task, content_type="metrics")
                # create new version task
                task.f_task_version = task.f_task_version + 1
                task.f_run_pid = None
                task.f_run_ip = None
                FederatedScheduler.create_task(job=job, task=task)
                # Save the status information of all participants in the initiator for scheduling
                schedule_logger(job_id=job_id).info(f"create task {task.f_task_id} new version {task.f_task_version}")
                for _role, _party_ids in job.f_runtime_conf["role"].items():
                    for _party_id in _party_ids:
                        if _role == initiator_role and _party_id == initiator_party_id:
                            continue
                        JobController.initialize_tasks(job_id, _role, _party_id, False, job.f_runtime_conf["initiator"], RunParameters(**job.f_runtime_conf["job_parameters"]), dsl_parser, component_name=task.f_component_name, task_version=task.f_task_version)
                schedule_logger(job_id=job_id).info(f"create task {task.f_task_id} new version {task.f_task_version} successfully")
                should_rerun = True
            else:
                schedule_logger(job_id=job_id).info(f"task {task.f_task_id} {task.f_task_version} on {task.f_role} {task.f_party_id} is complete, pass rerun")
        if should_rerun:
            if EndStatus.contains(job.f_status):
                job.f_status = JobStatus.WAITING
                job.f_end_time = None
                job.f_elapsed = None
                job.f_progress = 0
                schedule_logger(job_id=job_id).info(f"job {job_id} has been finished, set waiting to rerun")
                status, response = FederatedScheduler.sync_job_status(job=job)
                if status == FederatedSchedulingStatusCode.SUCCESS:
                    FederatedScheduler.sync_job(job=job, update_fields=["end_time", "elapsed", "progress"])
                    JobQueue.create_event(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
                    schedule_logger(job_id=job_id).info(f"job {job_id} set waiting to rerun successfully")
                else:
                    schedule_logger(job_id=job_id).info(f"job {job_id} set waiting to rerun failed")
            else:
                # status updates may be delayed, and in a very small probability they will be executed after the rerun command
                schedule_logger(job_id=job_id).info(f"job {job_id} status is {job.f_status}, will be run new version waiting task")
        else:
            schedule_logger(job_id=job_id).info(f"job {job_id} no task to rerun")

    @classmethod
    def update_job_on_initiator(cls, initiator_job: Job, update_fields: list):
        jobs = JobSaver.query_job(job_id=initiator_job.f_job_id)
        if not jobs:
            raise Exception("Failed to update job status on initiator")
        job_info = initiator_job.to_human_model_dict(only_primary_with=update_fields)
        for field in update_fields:
            job_info[field] = getattr(initiator_job, "f_%s" % field)
        for job in jobs:
            job_info["role"] = job.f_role
            job_info["party_id"] = job.f_party_id
            JobSaver.update_job_status(job_info=job_info)
            JobSaver.update_job(job_info=job_info)

    @classmethod
    def calculate_job_status(cls, task_scheduling_status_code, tasks_status):
        # 1. all waiting
        # 2. have running
        # 3. waiting + end status
        # 4. all end status and difference
        # 5. all the same end status
        tmp_status_set = set(tasks_status)
        if len(tmp_status_set) == 1:
            # 1 and 5
            return tmp_status_set.pop()
        else:
            if TaskStatus.RUNNING in tmp_status_set:
                # 2
                return JobStatus.RUNNING
            if TaskStatus.WAITING in tmp_status_set:
                # 3
                if task_scheduling_status_code == SchedulingStatusCode.HAVE_NEXT:
                    return JobStatus.RUNNING
                else:
                    pass
            # 3 with no next and 4
            for status in sorted(EndStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                if status == TaskStatus.COMPLETE:
                    continue
                elif status in tmp_status_set:
                    return status

            raise Exception("Calculate job status failed: {}".format(tasks_status))

    @classmethod
    def calculate_job_progress(cls, tasks_status):
        total = 0
        finished_count = 0
        for task_status in tasks_status:
            total += 1
            if EndStatus.contains(task_status):
                finished_count += 1
        return total, finished_count

    @classmethod
    def finish(cls, job, end_status):
        schedule_logger(job_id=job.f_job_id).info("Job {} finished with {}, do something...".format(job.f_job_id, end_status))
        FederatedScheduler.stop_job(job=job, stop_status=end_status)
        FederatedScheduler.clean_job(job=job)
        JobQueue.delete_event(job_id=job.f_job_id, initiator_role=job.f_initiator_role, initiator_party_id=job.f_initiator_party_id)
        schedule_logger(job_id=job.f_job_id).info("Job {} finished with {}, done".format(job.f_job_id, end_status))
