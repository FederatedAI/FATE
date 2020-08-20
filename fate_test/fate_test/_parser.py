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

import dataclasses
import json
import typing
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import click
import prettytable

from fate_test._config import Parties, Config


# noinspection PyPep8Naming
class chain_hook(object):
    def __init__(self):
        self._hooks = []

    def add_hook(self, hook):
        self._hooks.append(hook)
        return self

    def add_extend_namespace_hook(self, namespace):
        self.add_hook(_namespace_hook(namespace))
        return self

    def add_replace_hook(self, mapping):
        self.add_hook(_replace_hook(mapping))

    def hook(self, d):
        return self._chain_hooks(self._hooks, d)

    @staticmethod
    def _chain_hooks(hook_funcs, d):
        for hook_func in hook_funcs:
            if d is None:
                return
            d = hook_func(d)
        return d


DATA_JSON_HOOK = chain_hook()
CONF_JSON_HOOK = chain_hook()
DSL_JSON_HOOK = chain_hook()


@dataclass
class Data(object):
    config: dict
    role_str: str

    @staticmethod
    def load(config):
        kwargs = {}
        for field_name in ['file', 'head', 'partition', 'table_name', 'namespace']:
            kwargs[field_name] = config[field_name]
        role_str = config.get("role") if config.get("role") != "guest" else "guest_0"
        return Data(config=kwargs, role_str=role_str)


@dataclass
class JobConf(object):
    initiator: dict
    role: dict
    job_parameters: dict
    role_parameters: dict
    algorithm_parameters: dict = field(default_factory=dict)

    @staticmethod
    def load(path: Path):
        with path.open("r") as f:
            kwargs = json.load(f, object_hook=CONF_JSON_HOOK.hook)
        return JobConf(**kwargs)

    def update(self, parties: Parties, work_mode):
        self.initiator = parties.extract_initiator_role(self.initiator['role'])
        self.role = parties.extract_role({role: len(parties) for role, parties in self.role.items()})
        self.job_parameters.update(dict(work_mode=work_mode))


@dataclass
class JobDSL(object):
    components: dict

    @staticmethod
    def load(path: Path):
        with path.open("r") as f:
            kwargs = json.load(f, object_hook=DSL_JSON_HOOK.hook)
        return JobDSL(**kwargs)


@dataclass
class Job(object):
    job_name: str
    job_conf: JobConf
    job_dsl: typing.Optional[JobDSL]
    pre_works: typing.MutableSet[str]

    @classmethod
    def load(cls, job_name, job_configs, base: Path):
        job_conf = JobConf.load(base.joinpath(job_configs.get("conf")).resolve())
        job_dsl = job_configs.get("dsl", None)
        if job_dsl is not None:
            job_dsl = JobDSL.load(base.joinpath(job_dsl).resolve())

        pre_works = set()
        if job_configs.get("deps", None):
            pre_works.add(job_configs["deps"])
        return Job(job_name=job_name, job_conf=job_conf, job_dsl=job_dsl, pre_works=pre_works)

    @property
    def submit_params(self):
        return dict(conf=dataclasses.asdict(self.job_conf),
                    dsl=dataclasses.asdict(self.job_dsl) if self.job_dsl else None)

    def set_pre_work(self, name, **kwargs):
        if name not in self.pre_works:
            raise RuntimeError(f"{self} not dependents on {name} right now")
        self.job_conf.job_parameters.update(**kwargs)
        self.pre_works.remove(name)

    def is_submit_ready(self):
        return len(self.pre_works) == 0


@dataclass
class PipelineJob(object):
    job_name: str
    script_path: Path


@dataclass
class Testsuite(object):
    dataset: typing.List[Data]
    jobs: typing.List[Job]
    pipeline_jobs: typing.List[PipelineJob]
    path: Path

    @staticmethod
    def load(path: Path):
        with path.open("r") as f:
            testsuite_config = json.load(f, object_hook=DATA_JSON_HOOK.hook)

        dataset = []
        for d in testsuite_config.get("data"):
            dataset.append(Data.load(d))

        jobs = []
        for job_name, job_configs in testsuite_config.get("tasks", {}).items():
            jobs.append(Job.load(job_name=job_name, job_configs=job_configs, base=path.parent))

        pipeline_jobs = []
        for job_name, job_configs in testsuite_config.get("pipeline_tasks", {}).items():
            script_path = path.parent.joinpath(job_configs["script"]).resolve()
            pipeline_jobs.append(PipelineJob(job_name, script_path))

        testsuite = Testsuite(dataset, jobs, pipeline_jobs, path)
        return testsuite

    def jobs_iter(self):
        while self._ready_jobs:
            yield self._ready_jobs.pop()

    def pretty_final_summary(self):
        table = prettytable.PrettyTable(["job_name", "job_id", "status", "exception_id", "rest_dependency"])
        for status in self.get_final_status().values():
            table.add_row(
                [status.name, status.job_id, status.status, status.exception_id, ','.join(status.rest_dependency)])
        return table.get_string()

    def __post_init__(self):
        self._dependency: typing.MutableMapping[str, typing.List[Job]] = {}
        self._final_status: typing.MutableMapping[str, FinalStatus] = {}
        self._ready_jobs = deque()
        for job in self.jobs:
            for name in job.pre_works:
                self._dependency.setdefault(name, []).append(job)
            self._final_status[job.job_name] = FinalStatus(job.job_name)
            if job.is_submit_ready():
                self._ready_jobs.appendleft(job)

    def feed_success_model_info(self, name, model_info):
        if name not in self._dependency:
            return
        for job in self._dependency[name]:
            job.set_pre_work(name, **model_info)
            if job.is_submit_ready():
                self._ready_jobs.appendleft(job)
        del self._dependency[name]

    def reflash_configs(self, config: Config):

        for data in self.dataset:
            data.config.update(dict(work_mode=config.work_mode))

        for job in self.jobs:
            job.job_conf.update(config.parties, config.work_mode)
        return self

    def update_status(self, job_name, job_id: str = None, status: str = None, exception_id: str = None):
        for k, v in locals().items():
            if k != "job_name" and v is not None:
                setattr(self._final_status[job_name], k, v)

    def get_final_status(self):
        for name, jobs in self._dependency.items():
            for job in jobs:
                self._final_status[job.job_name].rest_dependency.append(name)
        return self._final_status


@dataclass
class FinalStatus(object):
    name: str
    job_id: str = "-"
    status: str = "not submitted"
    exception_id: str = "-"
    rest_dependency: typing.List[str] = field(default_factory=list)


def _namespace_hook(namespace):
    def _hook(d):
        if d is None:
            return d
        if 'namespace' in d and namespace:
            d['namespace'] = f"{d['namespace']}_{namespace}"
        return d

    return _hook


def _replace_hook(mapping: dict):
    def _hook(d):
        for k, v in mapping.items():
            if k in d:
                d[k] = v
        return d

    return _hook


class JsonParamType(click.ParamType):
    name = "json_string"

    def convert(self, value, param, ctx):
        try:
            return json.loads(value)
        except ValueError:
            self.fail(f"{value} is not a valid json string", param, ctx)


JSON_STRING = JsonParamType()
