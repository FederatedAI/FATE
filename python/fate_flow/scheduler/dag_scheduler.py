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

from fate_arch.common.base_utils import json_loads, json_dumps, current_timestamp
from fate_arch.common.log import schedule_logger
from fate_arch.common import WorkMode
from fate_flow.db.db_models import DB, Job
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.scheduler.task_scheduler import TaskScheduler
from fate_flow.operation.job_saver import JobSaver
from fate_flow.entity.types import JobStatus, TaskStatus, EndStatus, StatusSet, SchedulingStatusCode, ResourceOperation, \
    FederatedSchedulingStatusCode, RunParameters, RetCode
from fate_flow.operation.job_tracker import Tracker
from fate_flow.controller.job_controller import JobController
from fate_flow.utils import detect_utils, job_utils, schedule_utils, authentication_utils
from fate_flow.utils.config_adapter import JobRuntimeConfigAdapter
from fate_flow.utils import model_utils
from fate_flow.utils.cron import Cron
from fate_flow.settings import END_STATUS_JOB_SCHEDULING_TIME_LIMIT, END_STATUS_JOB_SCHEDULING_UPDATES


class DAGScheduler(Cron):
    @classmethod
    def submit(cls, job_data, job_id=None):
        if not job_id:
            job_id = job_utils.generate_job_id()
        schedule_logger(job_id).info('submit job, job_id {}, body {}'.format(job_id, job_data))
        job_dsl = job_data.get('job_dsl', {})
        job_runtime_conf = job_data.get('job_runtime_conf', {})
        job_utils.check_job_runtime_conf(job_runtime_conf)
        authentication_utils.check_constraint(job_runtime_conf, job_dsl)

        job_initiator = job_runtime_conf['initiator']
        conf_adapter = JobRuntimeConfigAdapter(job_runtime_conf)
        common_job_parameters = conf_adapter.get_common_parameters()

        if common_job_parameters.job_type != 'predict':
            # generate job model info
            common_job_parameters.model_id = model_utils.gen_model_id(job_runtime_conf['role'])
            common_job_parameters.model_version = job_id
            train_runtime_conf = {}
        else:
            # check predict job parameters
            detect_utils.check_config(common_job_parameters.to_dict(), ['model_id', 'model_version'])
            # get inference dsl from pipeline model as job dsl
            tracker = Tracker(job_id=job_id, role=job_initiator['role'], party_id=job_initiator['party_id'],
                              model_id=common_job_parameters.model_id, model_version=common_job_parameters.model_version)
            pipeline_model = tracker.get_output_model('pipeline')
            train_runtime_conf = json_loads(pipeline_model['Pipeline'].train_runtime_conf)
            if not model_utils.check_if_deployed(role=job_initiator['role'],
                                             party_id=job_initiator['party_id'],
                                             model_id=common_job_parameters.model_id,
                                             model_version=common_job_parameters.model_version):
                raise Exception(f"Model {common_job_parameters.model_id} {common_job_parameters.model_version} has not been deployed yet.")
            job_dsl = json_loads(pipeline_model['Pipeline'].inference_dsl)

        job = Job()
        job.f_job_id = job_id
        job.f_dsl = job_dsl
        job.f_train_runtime_conf = train_runtime_conf
        job.f_roles = job_runtime_conf['role']
        job.f_work_mode = common_job_parameters.work_mode
        job.f_initiator_role = job_initiator['role']
        job.f_initiator_party_id = job_initiator['party_id']
        job.f_role = job_initiator['role']
        job.f_party_id = job_initiator['party_id']

        path_dict = job_utils.save_job_conf(job_id=job_id,
                                            role=job.f_initiator_role,
                                            job_dsl=job_dsl,
                                            job_runtime_conf=job_runtime_conf,
                                            job_runtime_conf_on_party={},
                                            train_runtime_conf=train_runtime_conf,
                                            pipeline_dsl=None)

        if job.f_initiator_party_id not in job_runtime_conf['role'][job.f_initiator_role]:
            schedule_logger(job_id).info("initiator party id error:{}".format(job.f_initiator_party_id))
            raise Exception("initiator party id error {}".format(job.f_initiator_party_id))

        # create common parameters on initiator
        JobController.backend_compatibility(job_parameters=common_job_parameters)
        JobController.adapt_job_parameters(role=job.f_initiator_role, job_parameters=common_job_parameters, create_initiator_baseline=True)

        job.f_runtime_conf = conf_adapter.update_common_parameters(common_parameters=common_job_parameters)
        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job.f_dsl,
                                                       runtime_conf=job.f_runtime_conf,
                                                       train_runtime_conf=job.f_train_runtime_conf)

        # initiator runtime conf as template
        job.f_runtime_conf_on_party = job.f_runtime_conf.copy()
        job.f_runtime_conf_on_party["job_parameters"] = common_job_parameters.to_dict()

        #if common_job_parameters.work_mode == WorkMode.CLUSTER or common_job_parameters.backend :
        # Save the status information of all participants in the initiator for scheduling
        for role, party_ids in job.f_roles.items():
            for party_id in party_ids:
                if role == job.f_initiator_role and party_id == job.f_initiator_party_id:
                    continue
                JobController.initialize_tasks(job_id, role, party_id, False, job.f_initiator_role, job.f_initiator_party_id, common_job_parameters, dsl_parser)

        status_code, response = FederatedScheduler.create_job(job=job)
        if status_code != FederatedSchedulingStatusCode.SUCCESS:
            job.f_status = JobStatus.FAILED
            job.f_tag = "submit_failed"
            FederatedScheduler.sync_job_status(job=job)
            raise Exception("create job failed", response)

        schedule_logger(job_id).info(
            'submit job successfully, job id is {}, model id is {}'.format(job.f_job_id, common_job_parameters.model_id))
        logs_directory = job_utils.get_job_log_directory(job_id)
        submit_result = {
            "job_id": job_id,
            "model_info": {"model_id": common_job_parameters.model_id, "model_version": common_job_parameters.model_version},
            "logs_directory": logs_directory,
            "board_url": job_utils.get_board_url(job_id, job_initiator['role'], job_initiator['party_id'])
        }
        submit_result.update(path_dict)
        return submit_result

    def run_do(self):
        schedule_logger().info("start schedule waiting jobs")
        jobs = JobSaver.query_job(is_initiator=True, status=JobStatus.WAITING, order_by="create_time", reverse=False)
        schedule_logger().info(f"have {len(jobs)} waiting jobs")
        if len(jobs):
            # FIFO
            job = jobs[0]
            schedule_logger().info(f"schedule waiting job {job.f_job_id}")
            try:
                self.schedule_waiting_jobs(job=job)
            except Exception as e:
                schedule_logger(job.f_job_id).exception(e)
                schedule_logger(job.f_job_id).error(f"schedule waiting job {job.f_job_id} failed")
        schedule_logger().info("schedule waiting jobs finished")

        schedule_logger().info("start schedule running jobs")
        jobs = JobSaver.query_job(is_initiator=True, status=JobStatus.RUNNING, order_by="create_time", reverse=False)
        schedule_logger().info(f"have {len(jobs)} running jobs")
        for job in jobs:
            schedule_logger().info(f"schedule running job {job.f_job_id}")
            try:
                self.schedule_running_job(job=job)
            except Exception as e:
                schedule_logger(job.f_job_id).exception(e)
                schedule_logger(job.f_job_id).error(f"schedule job {job.f_job_id} failed")
        schedule_logger().info("schedule running jobs finished")

        # some ready job exit before start
        schedule_logger().info("start schedule ready jobs")
        jobs = JobSaver.query_job(is_initiator=True, ready_signal=True, order_by="create_time", reverse=False)
        schedule_logger().info(f"have {len(jobs)} ready jobs")
        for job in jobs:
            schedule_logger().info(f"schedule ready job {job.f_job_id}")
            try:
                self.schedule_ready_job(job=job)
            except Exception as e:
                schedule_logger(job.f_job_id).exception(e)
                schedule_logger(job.f_job_id).error(f"schedule ready job {job.f_job_id} failed:\n{e}")
        schedule_logger().info("schedule ready jobs finished")

        schedule_logger().info("start schedule rerun jobs")
        jobs = JobSaver.query_job(is_initiator=True, rerun_signal=True, order_by="create_time", reverse=False)
        schedule_logger().info(f"have {len(jobs)} rerun jobs")
        for job in jobs:
            schedule_logger().info(f"schedule rerun job {job.f_job_id}")
            try:
                self.schedule_rerun_job(job=job)
            except Exception as e:
                schedule_logger(job.f_job_id).exception(e)
                schedule_logger(job.f_job_id).error(f"schedule job {job.f_job_id} failed")
        schedule_logger().info("schedule rerun jobs finished")

        schedule_logger().info("start schedule end status jobs to update status")
        jobs = JobSaver.query_job(is_initiator=True, status=set(EndStatus.status_list()), end_time=[current_timestamp() - END_STATUS_JOB_SCHEDULING_TIME_LIMIT, current_timestamp()])
        schedule_logger().info(f"have {len(jobs)} end status jobs")
        for job in jobs:
            schedule_logger().info(f"schedule end status job {job.f_job_id}")
            try:
                update_status = self.end_scheduling_updates(job_id=job.f_job_id)
                if not update_status:
                    schedule_logger(job.f_job_id).info(f"the number of updates has been exceeded")
                    continue
                self.schedule_running_job(job=job)
            except Exception as e:
                schedule_logger(job.f_job_id).exception(e)
                schedule_logger(job.f_job_id).error(f"schedule job {job.f_job_id} failed")
        schedule_logger().info("schedule end status jobs finished")

    @classmethod
    def schedule_waiting_jobs(cls, job):
        job_id, initiator_role, initiator_party_id, = job.f_job_id, job.f_initiator_role, job.f_initiator_party_id,
        if not cls.ready_signal(job_id=job_id, set_or_reset=True):
            schedule_logger(job_id).info(f"job {job_id} may be handled by another scheduler")
            return
        try:
            if job.f_cancel_signal:
                job.f_status = JobStatus.CANCELED
                FederatedScheduler.sync_job_status(job=job)
                schedule_logger(job_id).info(f"job {job_id} have cancel signal")
                return
            apply_status_code, federated_response = FederatedScheduler.resource_for_job(job=job, operation_type=ResourceOperation.APPLY)
            if apply_status_code == FederatedSchedulingStatusCode.SUCCESS:
                cls.start_job(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
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
                    return_status_code, federated_response = FederatedScheduler.resource_for_job(job=job, operation_type=ResourceOperation.RETURN, specific_dest=rollback_party)
                    if return_status_code != FederatedSchedulingStatusCode.SUCCESS:
                        schedule_logger(job_id).info(f"job {job_id} return resource failed:\n{federated_response}")
                else:
                    schedule_logger(job_id).info(f"job {job_id} no party should be rollback resource")
                if apply_status_code == FederatedSchedulingStatusCode.ERROR:
                    cls.stop_job(job_id=job_id, role=initiator_role, party_id=initiator_party_id, stop_status=JobStatus.FAILED)
                    schedule_logger(job_id).info(f"apply resource error, stop job {job_id}")
        except Exception as e:
            raise e
        finally:
            update_status = cls.ready_signal(job_id=job_id, set_or_reset=False)
            schedule_logger(job_id).info(f"reset job {job_id} ready signal {update_status}")

    @classmethod
    def schedule_ready_job(cls, job):
        job_id, initiator_role, initiator_party_id, = job.f_job_id, job.f_initiator_role, job.f_initiator_party_id
        update_status = cls.ready_signal(job_id=job_id, set_or_reset=False, ready_timeout_ttl=60 * 1000)
        schedule_logger(job_id).info(f"reset job {job_id} ready signal {update_status}")

    @classmethod
    def schedule_rerun_job(cls, job):
        if EndStatus.contains(job.f_status):
            job.f_status = JobStatus.WAITING
            job.f_ready_signal = False
            job.f_ready_time = None
            job.f_rerun_signal = False
            job.f_progress = 0
            job.f_end_time = None
            job.f_elapsed = None
            schedule_logger(job_id=job.f_job_id).info(f"job {job.f_job_id} has been finished, set waiting to rerun")
            status, response = FederatedScheduler.sync_job_status(job=job)
            if status == FederatedSchedulingStatusCode.SUCCESS:
                cls.rerun_signal(job_id=job.f_job_id, set_or_reset=False)
                FederatedScheduler.sync_job(job=job, update_fields=["ready_signal", "ready_time", "rerun_signal", "progress", "end_time", "elapsed"])
                schedule_logger(job_id=job.f_job_id).info(f"job {job.f_job_id} set waiting to rerun successfully")
            else:
                schedule_logger(job_id=job.f_job_id).info(f"job {job.f_job_id} set waiting to rerun failed")
        else:
            cls.rerun_signal(job_id=job.f_job_id, set_or_reset=False)
            cls.schedule_running_job(job)

    @classmethod
    def start_job(cls, job_id, initiator_role, initiator_party_id):
        schedule_logger(job_id=job_id).info("try to start job {} on initiator {} {}".format(job_id, initiator_role, initiator_party_id))
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
                                                       runtime_conf=job.f_runtime_conf_on_party,
                                                       train_runtime_conf=job.f_train_runtime_conf)
        task_scheduling_status_code, tasks = TaskScheduler.schedule(job=job, dsl_parser=dsl_parser, canceled=job.f_cancel_signal)
        tasks_status = [task.f_status for task in tasks]
        new_job_status = cls.calculate_job_status(task_scheduling_status_code=task_scheduling_status_code, tasks_status=tasks_status)
        if new_job_status == JobStatus.WAITING and job.f_cancel_signal:
            new_job_status = JobStatus.CANCELED
        total, finished_count = cls.calculate_job_progress(tasks_status=tasks_status)
        new_progress = float(finished_count) / total * 100
        schedule_logger(job_id=job.f_job_id).info("Job {} status is {}, calculate by task status list: {}".format(job.f_job_id, new_job_status, tasks_status))
        if new_job_status != job.f_status or new_progress != job.f_progress:
            # Make sure to update separately, because these two fields update with anti-weight logic
            if int(new_progress) - job.f_progress > 0:
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
        job_can_rerun = False
        dsl_parser = schedule_utils.get_job_dsl_parser(dsl=job.f_dsl,
                                                       runtime_conf=job.f_runtime_conf_on_party,
                                                       train_runtime_conf=job.f_train_runtime_conf)
        for task in tasks:
            if task.f_status in {TaskStatus.WAITING, TaskStatus.SUCCESS}:
                if task.f_status == TaskStatus.WAITING:
                    job_can_rerun = True
                schedule_logger(job_id=job_id).info(f"task {task.f_task_id} {task.f_task_version} on {task.f_role} {task.f_party_id} is {task.f_status}, pass rerun")
            else:
                # stop old version task
                FederatedScheduler.stop_task(job=job, task=task, stop_status=TaskStatus.CANCELED)
                FederatedScheduler.clean_task(job=job, task=task, content_type="metrics")
                # create new version task
                task.f_task_version = task.f_task_version + 1
                task.f_run_pid = None
                task.f_run_ip = None
                FederatedScheduler.create_task(job=job, task=task)
                # Save the status information of all participants in the initiator for scheduling
                schedule_logger(job_id=job_id).info(f"create task {task.f_task_id} new version {task.f_task_version}")
                for _role, _party_ids in job.f_runtime_conf_on_party["role"].items():
                    for _party_id in _party_ids:
                        if _role == initiator_role and _party_id == initiator_party_id:
                            continue
                        JobController.initialize_tasks(job_id, _role, _party_id, False, job.f_initiator_role, job.f_initiator_party_id, RunParameters(**job.f_runtime_conf_on_party["job_parameters"]), dsl_parser, component_name=task.f_component_name, task_version=task.f_task_version)
                schedule_logger(job_id=job_id).info(f"create task {task.f_task_id} new version {task.f_task_version} successfully")
                job_can_rerun = True
        if job_can_rerun:
            schedule_logger(job_id=job_id).info(f"job {job_id} set rerun signal")
            status = cls.rerun_signal(job_id=job_id, set_or_reset=True)
            if status:
                schedule_logger(job_id=job_id).info(f"job {job_id} set rerun signal successfully")
            else:
                schedule_logger(job_id=job_id).info(f"job {job_id} set rerun signal failed")
        else:
            FederatedScheduler.sync_job_status(job=job)
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
                    # have waiting with no next
                    pass
            # have waiting with no next or 4
            for status in sorted(EndStatus.status_list(), key=lambda s: StatusSet.get_level(status=s), reverse=True):
                if status == TaskStatus.SUCCESS:
                    continue
                elif status in tmp_status_set:
                    return status
            if len(tmp_status_set) == 2 and TaskStatus.WAITING in tmp_status_set and TaskStatus.SUCCESS in tmp_status_set and task_scheduling_status_code == SchedulingStatusCode.NO_NEXT:
                return JobStatus.CANCELED

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
    def stop_job(cls, job_id, role, party_id, stop_status):
        schedule_logger(job_id=job_id).info(f"request stop job {job_id} with {stop_status}")
        jobs = JobSaver.query_job(job_id=job_id, role=role, party_id=party_id, is_initiator=True)
        if len(jobs) > 0:
            if stop_status == JobStatus.CANCELED:
                schedule_logger(job_id=job_id).info(f"cancel job {job_id}")
                set_cancel_status = cls.cancel_signal(job_id=job_id, set_or_reset=True)
                schedule_logger(job_id=job_id).info(f"set job {job_id} cancel signal {set_cancel_status}")
            job = jobs[0]
            job.f_status = stop_status
            schedule_logger(job_id=job_id).info(f"request stop job {job_id} with {stop_status} to all party")
            status_code, response = FederatedScheduler.stop_job(job=jobs[0], stop_status=stop_status)
            if status_code == FederatedSchedulingStatusCode.SUCCESS:
                schedule_logger(job_id=job_id).info(f"stop job {job_id} with {stop_status} successfully")
                return RetCode.SUCCESS, "success"
            else:
                initiator_tasks_group = JobSaver.get_tasks_asc(job_id=job.f_job_id, role=job.f_role, party_id=job.f_party_id)
                for initiator_task in initiator_tasks_group.values():
                    TaskScheduler.collect_task_of_all_party(job, initiator_task=initiator_task, set_status=stop_status)
                schedule_logger(job_id=job_id).info(f"stop job {job_id} with {stop_status} failed, {response}")
                return RetCode.FEDERATED_ERROR, json_dumps(response)
        else:
            return RetCode.SUCCESS, "can not found job"

    @classmethod
    @DB.connection_context()
    def ready_signal(cls, job_id, set_or_reset: bool, ready_timeout_ttl=None):
        filters = [Job.f_job_id == job_id]
        if set_or_reset:
            update_fields = {Job.f_ready_signal: True, Job.f_ready_time: current_timestamp()}
            filters.append(Job.f_ready_signal == False)
        else:
            update_fields = {Job.f_ready_signal: False, Job.f_ready_time: None}
            filters.append(Job.f_ready_signal == True)
            if ready_timeout_ttl:
                filters.append(current_timestamp() - Job.f_ready_time > ready_timeout_ttl)
        update_status = Job.update(update_fields).where(*filters).execute() > 0
        return update_status

    @classmethod
    @DB.connection_context()
    def cancel_signal(cls, job_id, set_or_reset: bool):
        update_status = Job.update({Job.f_cancel_signal: set_or_reset, Job.f_cancel_time: current_timestamp()}).where(Job.f_job_id == job_id).execute() > 0
        return update_status

    @classmethod
    @DB.connection_context()
    def rerun_signal(cls, job_id, set_or_reset: bool):
        if set_or_reset is True:
            update_fields = {Job.f_rerun_signal: True, Job.f_cancel_signal: False, Job.f_end_scheduling_updates: 0}
        elif set_or_reset is False:
            update_fields = {Job.f_rerun_signal: False}
        else:
            raise RuntimeError(f"can not support rereun signal {set_or_reset}")
        update_status = Job.update(update_fields).where(Job.f_job_id == job_id).execute() > 0
        return update_status

    @classmethod
    @DB.connection_context()
    def end_scheduling_updates(cls, job_id):
        operate = Job.update({Job.f_end_scheduling_updates: Job.f_end_scheduling_updates + 1}).where(Job.f_job_id == job_id,
                                                                                                     Job.f_end_scheduling_updates < END_STATUS_JOB_SCHEDULING_UPDATES)
        update_status = operate.execute() > 0
        return update_status

    @classmethod
    def finish(cls, job, end_status):
        schedule_logger(job_id=job.f_job_id).info("Job {} finished with {}, do something...".format(job.f_job_id, end_status))
        cls.stop_job(job_id=job.f_job_id, role=job.f_initiator_role, party_id=job.f_initiator_party_id, stop_status=end_status)
        FederatedScheduler.clean_job(job=job)
        schedule_logger(job_id=job.f_job_id).info("Job {} finished with {}, done".format(job.f_job_id, end_status))
