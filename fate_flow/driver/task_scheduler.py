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
import json
import os
import sys
import time
from threading import Timer

from arch.api.utils.core_utils import current_timestamp, base64_encode, json_loads, get_lan_ip
from arch.api.utils.log_utils import schedule_logger
from fate_flow.db.db_models import Job
from fate_flow.driver.task_executor import TaskExecutor
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.settings import API_VERSION, HTTP_PORT
from fate_flow.utils import job_utils
from fate_flow.utils.api_utils import federated_api
from fate_flow.utils.job_utils import query_task, get_job_dsl_parser, query_job
from fate_flow.entity.constant_config import JobStatus, Backend, TaskStatus


class TaskScheduler(object):
    @staticmethod
    def distribute_job(job, roles, job_initiator):
        for role, partys in roles.items():
            job.f_role = role
            for party_id in partys:
                job.f_party_id = party_id
                if role == job_initiator['role'] and party_id == job_initiator['party_id']:
                    job.f_is_initiator = 1
                else:
                    job.f_is_initiator = 0
                response_json = federated_api(job_id=job.f_job_id,
                                              method='POST',
                                              endpoint='/{}/schedule/{}/{}/{}/create'.format(
                                                  API_VERSION,
                                                  job.f_job_id,
                                                  role,
                                                  party_id),
                                              src_party_id=job_initiator['party_id'],
                                              dest_party_id=party_id, src_role=job_initiator['role'],
                                              json_body=job.to_json(),
                                              work_mode=job.f_work_mode)
                if response_json["retcode"]:
                    job.f_status = JobStatus.FAILED
                    TaskScheduler.sync_job_status(job_id=job.f_job_id, roles=roles,
                                                  work_mode=job.f_work_mode,
                                                  initiator_party_id=job_initiator['party_id'],
                                                  initiator_role=job_initiator['role'],
                                                  job_info=job.to_json())
                    raise Exception(
                        "an error occurred while creating the job: role {} party_id {}".format(role, party_id)
                        + "\n" + str(response_json["retmsg"]))

    @staticmethod
    def run_job(job_id, initiator_role, initiator_party_id):
        job_dsl, job_runtime_conf, train_runtime_conf = job_utils.get_job_configuration(job_id=job_id,
                                                                                        role=initiator_role,
                                                                                        party_id=initiator_party_id)
        job_parameters = job_runtime_conf.get('job_parameters', {})
        job_initiator = job_runtime_conf.get('initiator', {})
        dag = get_job_dsl_parser(dsl=job_dsl,
                                 runtime_conf=job_runtime_conf,
                                 train_runtime_conf=train_runtime_conf)
        job_args = dag.get_args_input()
        if not job_initiator:
            return False
        timeout = job_utils.get_timeout(job_id, job_parameters.get("timeout", None), job_runtime_conf, job_dsl)
        t = Timer(timeout, TaskScheduler.job_handler, [job_id])
        t.start()

        job = Job()
        job.f_job_id = job_id
        job.f_start_time = current_timestamp()
        job.f_status = JobStatus.RUNNING
        job.f_update_time = current_timestamp()
        TaskScheduler.sync_job_status(job_id=job_id, roles=job_runtime_conf['role'],
                                      work_mode=job_parameters['work_mode'],
                                      initiator_party_id=job_initiator['party_id'],
                                      initiator_role=job_initiator['role'],
                                      job_info=job.to_json())

        top_level_task_status = set()
        components = dag.get_next_components(None)
        schedule_logger(job_id).info(
            'job {} root components is {}'.format(job.f_job_id, [component.get_name() for component in components],
                                                  None))
        for component in components:
            try:
                # run a component as task
                run_status = TaskScheduler.run_component(job_id, job_runtime_conf, job_parameters, job_initiator,
                                                         job_args, dag,
                                                         component)
            except Exception as e:
                schedule_logger(job_id).exception(e)
                run_status = False
            top_level_task_status.add(run_status)
            if not run_status:
                break
        if len(top_level_task_status) == 2:
            job.f_status = JobStatus.FAILED
        elif True in top_level_task_status:
            job.f_status = JobStatus.COMPLETE
        else:
            job.f_status = JobStatus.FAILED
        job.f_end_time = current_timestamp()
        job.f_elapsed = job.f_end_time - job.f_start_time
        if job.f_status == JobStatus.COMPLETE:
            job.f_progress = 100
        job.f_update_time = current_timestamp()
        try:
            TaskScheduler.finish_job(job_id=job_id, job_runtime_conf=job_runtime_conf)
        except Exception as e:
            schedule_logger(job_id).exception(e)
            job.f_status = JobStatus.FAILED

        if job.f_status == JobStatus.FAILED:
            TaskScheduler.stop(job_id=job_id, end_status=JobStatus.FAILED)

        try:
            TaskScheduler.sync_job_status(job_id=job_id, roles=job_runtime_conf['role'],
                                          work_mode=job_parameters['work_mode'],
                                          initiator_party_id=job_initiator['party_id'],
                                          initiator_role=job_initiator['role'],
                                          job_info=job.to_json())
        except Exception as e:
            schedule_logger(job_id).exception(e)
            schedule_logger(job_id).warning('job {} sync status failed'.format(job.f_job_id))

        schedule_logger(job_id).info('job {} finished, status is {}'.format(job.f_job_id, job.f_status))
        t.cancel()


    @staticmethod
    def run_component(job_id, job_runtime_conf, job_parameters, job_initiator, job_args, dag, component):
        parameters = component.get_role_parameters()
        component_name = component.get_name()
        module_name = component.get_module()
        task_id = job_utils.generate_task_id(job_id=job_id, component_name=component_name)
        schedule_logger(job_id).info('job {} run component {}'.format(job_id, component_name))
        for role, partys_parameters in parameters.items():
            for party_index in range(len(partys_parameters)):
                party_parameters = partys_parameters[party_index]
                if role in job_args:
                    party_job_args = job_args[role][party_index]['args']
                else:
                    party_job_args = {}
                dest_party_id = party_parameters.get('local', {}).get('party_id')

                response = federated_api(job_id=job_id,
                              method='POST',
                              endpoint='/{}/schedule/{}/{}/{}/{}/{}/run'.format(
                                  API_VERSION,
                                  job_id,
                                  component_name,
                                  task_id,
                                  role,
                                  dest_party_id),
                              src_party_id=job_initiator['party_id'],
                              dest_party_id=dest_party_id,
                              src_role=job_initiator['role'],
                              json_body={'job_parameters': job_parameters,
                                         'job_initiator': job_initiator,
                                         'job_args': party_job_args,
                                         'parameters': party_parameters,
                                         'module_name': module_name,
                                         'input': component.get_input(),
                                         'output': component.get_output(),
                                         'job_server': {'ip': get_lan_ip(), 'http_port': RuntimeConfig.HTTP_PORT}},
                              work_mode=job_parameters['work_mode'])
                if response['retcode']:
                    if 'not authorized' in response['retmsg']:
                        raise Exception('run component {} not authorized'.format(component_name))
        component_task_status = TaskScheduler.check_task_status(job_id=job_id, component=component)
        job_status = TaskScheduler.check_job_status(job_id)
        if component_task_status and job_status:
            task_success = True
        else:
            task_success = False
        schedule_logger(job_id).info(
            'job {} component {} run {}'.format(job_id, component_name, 'success' if task_success else 'failed'))
        # update progress
        TaskScheduler.sync_job_status(job_id=job_id, roles=job_runtime_conf['role'],
                                      work_mode=job_parameters['work_mode'],
                                      initiator_party_id=job_initiator['party_id'],
                                      initiator_role=job_initiator['role'],
                                      job_info=job_utils.update_job_progress(job_id=job_id, dag=dag,
                                                                             current_task_id=task_id).to_json())
        TaskScheduler.stop(job_id=job_id, component_name=component_name)
        if task_success:
            next_components = dag.get_next_components(component_name)
            schedule_logger(job_id).info('job {} component {} next components is {}'.format(job_id, component_name,
                                                                                    [next_component.get_name() for
                                                                                     next_component in
                                                                                     next_components]))
            for next_component in next_components:
                try:
                    schedule_logger(job_id).info(
                        'job {} check component {} dependencies status'.format(job_id, next_component.get_name()))
                    dependencies_status = TaskScheduler.check_dependencies(job_id=job_id, dag=dag,
                                                                           component=next_component)
                    job_status = TaskScheduler.check_job_status(job_id)
                    schedule_logger(job_id).info(
                        'job {} component {} dependencies status is {}, job status is {}'.format(job_id, next_component.get_name(),
                                                                               dependencies_status, job_status))
                    if dependencies_status and job_status:
                        run_status = TaskScheduler.run_component(job_id, job_runtime_conf, job_parameters,
                                                                 job_initiator, job_args, dag,
                                                                 next_component)
                    else:
                        run_status = False
                except Exception as e:
                    schedule_logger(job_id).exception(e)
                    run_status = False
                if not run_status:
                    return False
            return True
        else:
            if component_task_status == None:
                end_status = JobStatus.TIMEOUT
            else:
                end_status = JobStatus.FAILED
            TaskScheduler.stop(job_id=job_id, end_status=end_status)
            return False

    @staticmethod
    def check_dependencies(job_id, dag, component):
        role, party_id = job_utils.query_job_info(job_id)
        dependencies = dag.get_dependency(role=role, party_id=int(party_id)).get('dependencies', {})
        if not dependencies:
            return False
        dependent_component_names = dependencies.get(component.get_name(), [])
        schedule_logger(job_id).info('job {} component {} all dependent component: {}'.format(job_id, component.get_name(),
                                                                                      dependent_component_names))
        for dependent_component in dependent_component_names:
            dependent_component_name = dependent_component["component_name"]
            dependent_component = dag.get_component_info(dependent_component_name)
            dependent_component_task_status = TaskScheduler.check_task_status(job_id, dependent_component)
            schedule_logger(job_id).info('job {} component {} dependency {} status is {}'.format(job_id, component.get_name(),
                                                                                         dependent_component_name,
                                                                                         dependent_component_task_status))
            if not dependent_component_task_status:
                # dependency component run failed, break
                return False
        else:
            return True

    @staticmethod
    def check_task_status(job_id, component, interval=0.5):
        task_id = job_utils.generate_task_id(job_id=job_id, component_name=component.get_name())
        while True:
            try:
                status_collect = set()
                parameters = component.get_role_parameters()
                for _role, _partys_parameters in parameters.items():
                    for _party_parameters in _partys_parameters:
                        _party_id = _party_parameters.get('local', {}).get('party_id')
                        tasks = query_task(job_id=job_id, task_id=task_id, role=_role, party_id=_party_id)
                        if tasks:
                            task_status = tasks[0].f_status
                        else:
                            task_status = 'notRunning'
                        schedule_logger(job_id).info(
                            'job {} component {} run on {} {} status is {}'.format(job_id, component.get_name(), _role,
                                                                                   _party_id, task_status))
                        status_collect.add(task_status)
                if 'failed' in status_collect:
                    return False
                if 'timeout' in status_collect:
                    return None
                elif len(status_collect) == 1 and TaskStatus.COMPLETE in status_collect:
                    return True
                else:
                    time.sleep(interval)
            except Exception as e:
                schedule_logger(job_id).exception(e)
                return False

    @staticmethod
    def check_job_status(job_id):
        jobs = query_job(job_id=job_id)
        for job in jobs:
            if job.f_status != JobStatus.RUNNING:
                return False
        return True

    @staticmethod
    def run_task(job_id, component_name, task_id, role, party_id, task_config):
        schedule_logger(job_id).info(
            'job {} {} {} {} task subprocess is ready'.format(job_id, component_name, role, party_id, task_config))
        task_process_start_status = False
        try:
            task_dir = os.path.join(job_utils.get_job_directory(job_id=job_id), role, party_id, component_name)
            os.makedirs(task_dir, exist_ok=True)
            task_config_path = os.path.join(task_dir, 'task_config.json')
            with open(task_config_path, 'w') as fw:
                json.dump(task_config, fw)

            try:
                backend = task_config['job_parameters']['backend']
            except KeyError:
                backend = 0
                schedule_logger(job_id).warning("failed to get backend, set as 0")

            backend = Backend(backend)

            if backend.is_eggroll():
                process_cmd = [
                    'python3', sys.modules[TaskExecutor.__module__].__file__,
                    '-j', job_id,
                    '-n', component_name,
                    '-t', task_id,
                    '-r', role,
                    '-p', party_id,
                    '-c', task_config_path,
                    '--processors_per_node', str(task_config['job_parameters'].get("processors_per_node", 0)),
                    '--job_server', '{}:{}'.format(get_lan_ip(), HTTP_PORT),
                ]
            elif backend.is_spark():
                if "SPARK_HOME" not in os.environ:
                    raise EnvironmentError("SPARK_HOME not found")
                spark_home = os.environ["SPARK_HOME"]

                # additional configs
                spark_submit_config = task_config['job_parameters'].get("spark_submit_config", dict())

                deploy_mode = spark_submit_config.get("deploy-mode", "client")
                if deploy_mode not in ["client"]:
                    raise ValueError(f"deploy mode {deploy_mode} not supported")

                spark_submit_cmd = os.path.join(spark_home, "bin/spark-submit")
                process_cmd = [spark_submit_cmd, f'--name={task_id}#{role}']
                for k, v in spark_submit_config.items():
                    if k != "conf":
                        process_cmd.append(f'--{k}={v}')
                if "conf" in spark_submit_config:
                    for ck, cv in spark_submit_config["conf"].items():
                        process_cmd.append(f'--conf')
                        process_cmd.append(f'{ck}={cv}')
                process_cmd.extend([
                    sys.modules[TaskExecutor.__module__].__file__,
                    '-j', job_id,
                    '-n', component_name,
                    '-t', task_id,
                    '-r', role,
                    '-p', party_id,
                    '-c', task_config_path,
                    '--job_server',
                    '{}:{}'.format(get_lan_ip(), HTTP_PORT),
                ])
            else:
                raise ValueError(f"${backend} supported")

            task_log_dir = os.path.join(job_utils.get_job_log_directory(job_id=job_id), role, party_id, component_name)
            schedule_logger(job_id).info(
                'job {} {} {} {} task subprocess start'.format(job_id, component_name, role, party_id, task_config))
            p = job_utils.run_subprocess(config_dir=task_dir, process_cmd=process_cmd, log_dir=task_log_dir)
            if p:
                task_process_start_status = True
        except Exception as e:
            schedule_logger(job_id).exception(e)
        finally:
            schedule_logger(job_id).info(
                'job {} component {} on {} {} start task subprocess {}'.format(job_id, component_name, role, party_id,
                                                                               'success' if task_process_start_status else 'failed'))

    @staticmethod
    def sync_job_status(job_id, roles, work_mode, initiator_party_id, initiator_role, job_info):
        for role, partys in roles.items():
            job_info['f_role'] = role
            for party_id in partys:
                job_info['f_party_id'] = party_id
                federated_api(job_id=job_id,
                              method='POST',
                              endpoint='/{}/schedule/{}/{}/{}/status'.format(
                                  API_VERSION,
                                  job_id,
                                  role,
                                  party_id),
                              src_party_id=initiator_party_id,
                              dest_party_id=party_id,
                              src_role=initiator_role,
                              json_body=job_info,
                              work_mode=work_mode)

    @staticmethod
    def finish_job(job_id, job_runtime_conf, stop=False):
        job_parameters = job_runtime_conf['job_parameters']
        job_initiator = job_runtime_conf['initiator']
        model_id_base64 = base64_encode(job_parameters['model_id'])
        model_version_base64 = base64_encode(job_parameters['model_version'])
        roles = ','.join(job_runtime_conf['role'].keys())
        party_ids = ','.join([','.join([str(j) for j in i]) for i in job_runtime_conf['role'].values()])
        for role, partys in job_runtime_conf['role'].items():
            for party_id in partys:
                # save pipeline
                if not stop:
                    federated_api(job_id=job_id,
                                  method='POST',
                                  endpoint='/{}/schedule/{}/{}/{}/{}/{}/save/pipeline'.format(
                                      API_VERSION,
                                      job_id,
                                      role,
                                      party_id,
                                      model_id_base64,
                                      model_version_base64
                                  ),
                                  src_party_id=job_initiator['party_id'],
                                  dest_party_id=party_id,
                                  src_role=job_initiator['role'],
                                  json_body={},
                                  work_mode=job_parameters['work_mode'])
                # clean
                federated_api(job_id=job_id,
                              method='POST',
                              endpoint='/{}/schedule/{}/{}/{}/{}/{}/clean'.format(
                                  API_VERSION,
                                  job_id,
                                  role,
                                  party_id,
                                  roles,
                                  party_ids
                              ),
                              src_party_id=job_initiator['party_id'],
                              dest_party_id=party_id,
                              src_role=job_initiator['role'],
                              json_body={},
                              work_mode=job_parameters['work_mode'])
        schedule_logger(job_id, delete=True)

    @staticmethod
    def stop(job_id, end_status=JobStatus.FAILED, component_name=''):
        schedule_logger(job_id).info('get {} job {} {} command'.format("cancel" if end_status == JobStatus.CANCELED else "stop", job_id, component_name))
        jobs = job_utils.query_job(job_id=job_id, is_initiator=1)
        cancel_success = False
        is_cancel = (end_status == JobStatus.CANCELED)
        if jobs:
            initiator_job = jobs[0]
            job_info = {'f_job_id': job_id, 'f_status': end_status}
            roles = json_loads(initiator_job.f_roles)
            job_work_mode = initiator_job.f_work_mode
            initiator_party_id = initiator_job.f_party_id

            # set status first
            if not component_name:
                TaskScheduler.sync_job_status(job_id=job_id, roles=roles, initiator_party_id=initiator_party_id,
                                              initiator_role=initiator_job.f_role,
                                              work_mode=job_work_mode,
                                              job_info=job_info)
            for role, partys in roles.items():
                for party_id in partys:
                    response = federated_api(job_id=job_id,
                                             method='POST',
                                             endpoint='/{}/schedule/{}/{}/{}/{}'.format(
                                                 API_VERSION,
                                                 job_id,
                                                 role,
                                                 party_id,
                                                 "cancel" if is_cancel else "kill"
                                             ),
                                             src_party_id=initiator_party_id,
                                             dest_party_id=party_id,
                                             src_role=initiator_job.f_role,
                                             json_body={'job_initiator': {'party_id': initiator_job.f_party_id,
                                                                          'role': initiator_job.f_role},
                                                        'timeout': end_status == JobStatus.TIMEOUT,
                                                        'component_name': component_name
                                                        },
                                             work_mode=job_work_mode)
                    if response['retcode'] == 0:
                        cancel_success = True
                        schedule_logger(job_id).info(
                            'send {} {} {} job {} {} command successfully'.format(role, party_id, "cancel" if is_cancel else "kill",
                                                                                  job_id, component_name))
                        if is_cancel:
                            break
                    else:
                        schedule_logger(job_id).info(
                            'send {} {} {} job {} {} command failed: {}'.format(role, party_id, "cancel" if is_cancel else "kill",
                                                                                job_id, component_name, response['retmsg']))
            if is_cancel:
                return cancel_success
        else:
            jobs = job_utils.query_job(job_id=job_id)
            if jobs:
                raise Exception('Current role is not this job initiator')
            schedule_logger(job_id).info('send {} job {} {} command failed'.format("cancel" if is_cancel else "kill", job_id, component_name))
            raise Exception('can not found job: {}'.format(job_id))

    @staticmethod
    def clean_queue():
        schedule_logger().info('get clean queue command')
        jobs = job_utils.query_job(is_initiator=1, status=JobStatus.WAITING)
        if jobs:
            for job in jobs:
                schedule_logger(job.f_job_id).info(
                    'start send {} job {} command success'.format(JobStatus.CANCELED, job.f_job_id))
                job_info = {'f_job_id': job.f_job_id, 'f_status': JobStatus.CANCELED}
                roles = json_loads(job.f_roles)
                job_work_mode = job.f_work_mode
                initiator_party_id = job.f_party_id

                TaskScheduler.sync_job_status(job_id=job.f_job_id, roles=roles, initiator_party_id=initiator_party_id,
                                              initiator_role=job.f_role,
                                              work_mode=job_work_mode,
                                              job_info=job_info)
                job_runtime_conf = json_loads(job.f_runtime_conf)
                event = job_utils.job_event(job.f_job_id,
                                            job_runtime_conf['initiator']['role'],
                                            job_runtime_conf['initiator']['party_id'])
                try:
                    RuntimeConfig.JOB_QUEUE.del_event(event)
                    schedule_logger(job.f_job_id).info(
                        'send {} job {} command success'.format(JobStatus.CANCELED, job.f_job_id))
                except Exception as e:
                    schedule_logger(job.f_job_id).error(e)
        else:
            raise Exception('There are no jobs in the queue')

    @staticmethod
    def job_handler(*args, **kwargs):
        job_id = args[0]
        schedule_logger(job_id).info('job {} running timeout, start stop job'.format(job_id))
        TaskScheduler.stop(job_id, end_status=JobStatus.TIMEOUT)