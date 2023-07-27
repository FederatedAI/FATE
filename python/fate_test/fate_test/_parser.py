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

import typing
from pathlib import Path

import prettytable
from fate_test._config import Config
from fate_test._io import echo
from fate_test.utils import TxtStyle
# import json
from ruamel import yaml

from fate_test import _config


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


class Data(object):
    def __init__(self, config: dict, role_str: str):
        self.config = config
        self.file = config.get("file", "")
        self.meta = config.get("meta", {})
        self.partitions = config.get("partitions", 4)
        self.head = config.get("head", True)
        self.extend_sid = config.get("extend_sid", True)
        self.namespace = config.get("namespace", "")
        self.table_name = config.get("table_name", "")
        self.role_str = role_str

    @staticmethod
    def load(config, path: Path):
        kwargs = {}
        for field_name in config.keys():
            if field_name not in ["file", "role"]:
                kwargs[field_name] = config[field_name]
        # if config.get("engine", {}) != "PATH":
        file_path = path.parent.joinpath(config["file"]).resolve()
        if not file_path.exists():
            kwargs["file"] = config["file"]
        else:
            kwargs["file"] = file_path
        role_str = config.get("role") if config.get("role") != "guest" else "guest_0"
        return Data(config=kwargs, role_str=role_str)

    def update(self, config: Config):
        if config.extend_sid is not None:
            self.extend_sid = config.extend_sid
        """if config.meta is not None:
            self.meta.update(config.meta)"""


class PipelineJob(object):
    def __init__(self, job_name: str, script_path: Path):
        self.job_name = job_name
        self.script_path = script_path


class Testsuite(object):
    def __init__(
            self,
            dataset: typing.List[Data],
            # jobs: typing.List[Job],
            pipeline_jobs: typing.List[PipelineJob],
            path: Path,
    ):
        self.dataset = dataset
        # self.jobs = jobs
        self.pipeline_jobs = pipeline_jobs
        self.path = path
        self.suite_name = Path(self.path).stem
        self._final_status: typing.MutableMapping[str, FinalStatus] = {}
        """
        self._dependency: typing.MutableMapping[str, typing.List[Job]] = {}
        self._ready_jobs = deque()
        for job in self.jobs:
            for name in job.pre_works:
                self._dependency.setdefault(name, []).append(job)

            self._final_status[job.job_name] = FinalStatus(job.job_name)
            if job.is_submit_ready():
                self._ready_jobs.appendleft(job)"""

        for job in self.pipeline_jobs:
            self._final_status[job.job_name] = FinalStatus(job.job_name)

    @staticmethod
    def load(path: Path, provider):
        with path.open("r") as f:
            # testsuite_config = json.load(f, object_hook=DATA_JSON_HOOK.hook)
            testsuite_config = yaml.safe_load(f)
            # testsuite_config = DATA_JSON_HOOK.hook(testsuite_config)

        dataset = []
        for d in testsuite_config.get("data"):
            d = DATA_JSON_HOOK.hook(d)
            """if "use_local_data" not in d:
                d.update({"use_local_data": _config.use_local_data})"""
            dataset.append(Data.load(d, path))

        pipeline_jobs = []
        if testsuite_config.get("tasks", None) is not None and provider is not None:
            echo.echo('[Warning]  Pipeline does not support parameter: provider-> {}'.format(provider))
        for job_name, job_configs in testsuite_config.get("tasks", {}).items():
            script_path = path.parent.joinpath(job_configs["script"]).resolve()
            pipeline_jobs.append(PipelineJob(job_name, script_path))

        testsuite = Testsuite(dataset, pipeline_jobs, path)
        return testsuite

    """def jobs_iter(self) -> typing.Generator[Job, None, None]:
        while self._ready_jobs:
            yield self._ready_jobs.pop()"""

    @staticmethod
    def style_table(txt):
        colored_txt = txt.replace("success", f"{TxtStyle.TRUE_VAL}success{TxtStyle.END}")
        colored_txt = colored_txt.replace("failed", f"{TxtStyle.FALSE_VAL}failed{TxtStyle.END}")
        colored_txt = colored_txt.replace("not submitted", f"{TxtStyle.FALSE_VAL}not submitted{TxtStyle.END}")
        return colored_txt

    def pretty_final_summary(self, time_consuming, suite_file=None):
        """table = prettytable.PrettyTable(
            ["job_name", "job_id", "status", "time_consuming", "exception_id", "rest_dependency"]
        )"""
        table = prettytable.PrettyTable()
        table.set_style(prettytable.ORGMODE)
        field_names = ["job_name", "job_id", "status", "time_consuming", "exception_id", "rest_dependency"]
        table.field_names = field_names
        for status in self.get_final_status().values():
            if status.status != "success":
                status.suite_file = suite_file
                _config.non_success_jobs.append(status)
            if status.exception_id != "-":
                exception_id_txt = f"{TxtStyle.FALSE_VAL}{status.exception_id}{TxtStyle.END}"
            else:
                exception_id_txt = f"{TxtStyle.FIELD_VAL}{status.exception_id}{TxtStyle.END}"
            table.add_row(
                [
                    f"{TxtStyle.FIELD_VAL}{status.name}{TxtStyle.END}",
                    f"{TxtStyle.FIELD_VAL}{status.job_id}{TxtStyle.END}",
                    self.style_table(status.status),
                    f"{TxtStyle.FIELD_VAL}{time_consuming.pop(0) if status.job_id != '-' else '-'}{TxtStyle.END}",
                    f"{exception_id_txt}",
                    f"{TxtStyle.FIELD_VAL}{','.join(status.rest_dependency)}{TxtStyle.END}",
                ]
            )

        return table.get_string(title=f"{TxtStyle.TITLE}Testsuite Summary: {self.suite_name}{TxtStyle.END}")

    def model_in_dep(self, name):
        return name in self._dependency

    """def get_dependent_jobs(self, name):
        return self._dependency[name]"""

    def remove_dependency(self, name):
        del self._dependency[name]

    """def feed_dep_info(self, job, name, model_info=None, table_info=None, cache_info=None, model_loader_info=None):
        if model_info is not None:
            job.set_pre_work(name, **model_info)
        if table_info is not None:
            job.set_input_data(table_info["hierarchy"], table_info["table_info"])
        if cache_info is not None:
            job.set_input_data(cache_info["hierarchy"], cache_info["cache_info"])
        if model_loader_info is not None:
            job.set_input_data(model_loader_info["hierarchy"], model_loader_info["model_loader_info"])
        if name in job.pre_works:
            job.pre_works.remove(name)
        if job.is_submit_ready():
            self._ready_jobs.appendleft(job)"""

    def reflash_configs(self, config: Config):
        failed = []
        for job in self.jobs:
            try:
                job.job_conf.update(
                    config.parties, None, {}, {}
                )
            except ValueError as e:
                failed.append((job, e))
        return failed

    def update_status(
            self, job_name, job_id: str = None, status: str = None, exception_id: str = None
    ):
        for k, v in locals().items():
            if k != "job_name" and v is not None:
                setattr(self._final_status[job_name], k, v)

    def get_final_status(self):
        """for name, jobs in self._dependency.items():
            for job in jobs:
                self._final_status[job.job_name].rest_dependency.append(name)"""
        return self._final_status


class FinalStatus(object):
    def __init__(
            self,
            name: str,
            job_id: str = "-",
            status: str = "not submitted",
            exception_id: str = "-",
            rest_dependency: typing.List[str] = None,
    ):
        self.name = name
        self.job_id = job_id
        self.status = status
        self.exception_id = exception_id
        self.rest_dependency = rest_dependency or []
        self.suite_file = None


class BenchmarkJob(object):
    def __init__(self, job_name: str, script_path: Path, conf_path: Path):
        self.job_name = job_name
        self.script_path = script_path
        self.conf_path = conf_path


class BenchmarkPair(object):
    def __init__(
            self, pair_name: str, jobs: typing.List[BenchmarkJob], compare_setting: dict
    ):
        self.pair_name = pair_name
        self.jobs = jobs
        self.compare_setting = compare_setting


class BenchmarkSuite(object):
    def __init__(
            self, dataset: typing.List[Data], pairs: typing.List[BenchmarkPair], path: Path
    ):
        self.dataset = dataset
        self.pairs = pairs
        self.path = path

    @staticmethod
    def load(path: Path):
        with path.open("r") as f:
            # testsuite_config = json.load(f, object_hook=DATA_JSON_HOOK.hook)
            testsuite_config = yaml.safe_load(f)
            # testsuite_config = DATA_JSON_HOOK.hook(testsuite_config)

        dataset = []
        for d in testsuite_config.get("data"):
            d = DATA_JSON_HOOK.hook(d)
            dataset.append(Data.load(d, path))

        pairs = []
        for pair_name, pair_configs in testsuite_config.items():
            if pair_name == "data":
                continue
            jobs = []
            for job_name, job_configs in pair_configs.items():
                if job_name == "compare_setting":
                    continue
                script_path = path.parent.joinpath(job_configs["script"]).resolve()
                if job_configs.get("conf"):
                    conf_path = path.parent.joinpath(job_configs["conf"]).resolve()
                else:
                    conf_path = ""
                jobs.append(
                    BenchmarkJob(
                        job_name=job_name, script_path=script_path, conf_path=conf_path
                    )
                )
            compare_setting = pair_configs.get("compare_setting")
            if compare_setting and not isinstance(compare_setting, dict):
                raise ValueError(
                    f"expected 'compare_setting' type is dict, received {type(compare_setting)} instead."
                )
            pairs.append(
                BenchmarkPair(
                    pair_name=pair_name, jobs=jobs, compare_setting=compare_setting
                )
            )
        suite = BenchmarkSuite(dataset=dataset, pairs=pairs, path=path)
        return suite


def non_success_summary():
    status = {}
    for job in _config.non_success_jobs:
        if job.status not in status.keys():
            status[job.status] = prettytable.PrettyTable(
                ["testsuite_name", "job_name", "job_id", "status", "exception_id", "rest_dependency"]
            )

        status[job.status].add_row(
            [
                job.suite_file,
                job.name,
                job.job_id,
                job.status,
                job.exception_id,
                ",".join(job.rest_dependency),
            ]
        )
    for k, v in status.items():
        echo.echo("\n" + "#" * 60)
        echo.echo(v.get_string(title=f"{k} job record"), fg='red')


def _namespace_hook(namespace):
    def _hook(d):
        if d is None:
            return d
        if "namespace" in d and namespace:
            d["namespace"] = f"{d['namespace']}_{namespace}"
        return d

    return _hook


def _replace_hook(mapping: dict):
    def _hook(d):
        for k, v in mapping.items():
            if k in d:
                d[k] = v
        return d

    return _hook


"""class JsonParamType(click.ParamType):
    name = "json_string"

    def convert(self, value, param, ctx):
        try:
            return json.loads(value)
        except ValueError:
            self.fail(f"{value} is not a valid json string", param, ctx)


JSON_STRING = JsonParamType()"""
