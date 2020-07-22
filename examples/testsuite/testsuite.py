import abc
import argparse
import json
import pprint
import sys
import tempfile
import time
import typing
from enum import Enum
from pathlib import Path

import loguru
import prettytable
import sshtunnel
import yaml

LOGGER = loguru.logger


def main():
    parser = argparse.ArgumentParser("TESTSUITE RUNNER")
    parser.add_argument("path", help="path to search xxx_testsuite.json")
    parser.add_argument("-drop", default=1, type=int, choices=[0, 1],
                        help="whether to drop the table if table already exists, 1 for yes and 0 for no")
    parser.add_argument("-config", default="./config.yaml", type=str,
                        help="config file")
    parser.add_argument("-name", default=f'testsuite-{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    parser.add_argument("-work_mode", choices=[0, 1], type=int)
    parser.add_argument("-client", choices=["flowpy"], type=str)
    parser.add_argument("-replace", type=str)
    args = parser.parse_args()
    _add_logger(args.name)
    path = Path(args.path)
    config_overwrite = {}
    if args.work_mode is not None:
        config_overwrite["work_mode"] = args.work_mode
    if args.client is not None:
        config_overwrite["client"] = args.client

    hook = None
    if args.replace is not None:
        hook = _replace_hook(json.loads(args.replace))

    with Clients(config_path=Path(args.config), drop=args.drop, **config_overwrite) as clients:
        if path.is_file():
            if path.name.endswith("testsuite.json"):
                paths = [path]
            else:
                LOGGER.warning(f"{path} is file, but not end with `testsuite.json`, skip")
                return
        else:
            paths = path.glob(f"**/*testsuite.json")
        paths = [path.resolve() for path in paths]
        testsuites = {path.__str__(): _TestSuite.load(path, hook=hook) for path in paths}
        clients.run_testsuites(testsuites)


def _replace_hook(mapping: dict):
    def _hook(d):
        for k, v in mapping.items():
            if k in d:
                d[k] = v
        return d

    return _hook


def _add_logger(name):
    global LOGGER
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, level="INFO", colorize=True)
    loguru.logger.add(f"{name}-info.log", level="INFO")
    loguru.logger.add(f"{name}-debug.log", level="DEBUG")
    LOGGER = loguru.logger


class Clients(object):
    def __init__(self, config_path: Path, **kwargs):
        with config_path.open() as f:
            conf = yaml.load(f)
        if kwargs:
            conf.update(kwargs)

        LOGGER.debug(f"config: {pprint.pformat(conf)}")

        self._role_to_parties = conf.get("parties")
        parties_to_role_string = {}
        for role, parties in self._role_to_parties.items():
            for i, party in enumerate(parties):
                parties_to_role_string[party] = f"{role.lower()}_{i}"

        tunnels = []
        for ssh_conf in conf.get("ssh_tunnel"):
            ssh_address = ssh_conf.get("ssh_address")
            ssh_host, ssh_port = _parse_address(ssh_address)
            ssh_username = ssh_conf.get("ssh_username")
            ssh_password = ssh_conf.get("ssh_password")
            ssh_pkey = ssh_conf.get("ssh_priv_key")
            services = ssh_conf.get("services")

            role_strings = []
            remote_bind_addresses = []
            for service in services:
                role_strings.append([parties_to_role_string[party] for party in service.get("parties")])
                remote_bind_addresses.append(_parse_address(service.get("address")))

            tunnel = sshtunnel.SSHTunnelForwarder(ssh_address_or_host=(ssh_host, ssh_port),
                                                  ssh_username=ssh_username,
                                                  ssh_password=ssh_password,
                                                  ssh_pkey=ssh_pkey,
                                                  remote_bind_addresses=remote_bind_addresses)
            tunnels.append((tunnel, role_strings))

        self._client_type = conf.get("client", "flowpy").lower()
        self._drop = conf.get("drop", 0)
        self._work_mode = int(conf.get("work_mode", "0"))
        self._tunnels: typing.List[typing.Tuple[sshtunnel.SSHTunnelForwarder, typing.List[typing.List[str]]]] = tunnels
        self._clients: typing.MutableMapping[str, _Client] = {}
        self._default_client = _client_factory(self._client_type)

    def run_testsuite(self, testsuite: '_TestSuite'):
        num_data = len(testsuite.data)
        LOGGER.info(f"num of data to upload: {num_data}")

        # upload data, raise exception if any exception occurs or data upload job failed.
        for i, data in enumerate(testsuite.data):
            LOGGER.info(f"uploading ({i + 1}/{num_data})")
            LOGGER.debug(f"uploading data: {data}")
            client = self._get_client(data.role_str)

            # submit job
            job_id = client.upload_data(data.as_dict(work_mode=self._work_mode), drop=self._drop)
            LOGGER.opt(colors=True).info(f"submitted, job id: <red>{job_id}</red>")

            # check status
            status = client.query_job(job_id=job_id, role="local")
            LOGGER.opt(colors=True).info(f"uploaded, status: <green>{status}</green>")
            if status != "success":
                raise Exception(f"upload {i + 1}th data failed")

        # submit jobs, jobs's exception will logged then ignored
        num_task = len(testsuite.task)
        LOGGER.info(f"num of task to submit: {num_task}")
        deps_info = {}
        summary = []
        for i, task in enumerate(testsuite.task):
            task_summary = {"name": task.name}
            try:
                start = time.time()
                LOGGER.info(f"submitting {task.name} ({i + 1}/{num_task})")
                LOGGER.debug(f"submitting task:\n{task}")
                client = self._get_client("guest_0")

                # preprocess, modify conf
                task.update_roles(self._role_to_parties)
                task.update_job_parameters(work_mode=self._work_mode)
                task.update_deps(deps_info)
                LOGGER.debug(f"task modified:\n{task}")

                # submit job
                job_id, model_info = client.submit_job(task.conf, task.dsl)
                task_summary["job_id"] = job_id
                LOGGER.opt(colors=True).info(f"submitted, job_id: <red>{job_id}</red>")

                # check status, block until job is completed
                status = client.query_job(job_id=job_id, role="guest")
                task_summary["status"] = status
                if status == "success":
                    deps_info[task.name] = model_info
                LOGGER.opt(colors=True).info(f"job completed, status: <green>{status}</green>,"
                                             f"takes {time.time() - start}s")
            except Exception as e:
                LOGGER.exception(f"task {task.name} error, {e}")
                task_summary["status"] = "exception"
            summary.append(task_summary)

        #  testsuite summery
        LOGGER.info(f"\n{_pretty_summary(summary)}")
        return summary

    def run_testsuites(self, testsuites: typing.MutableMapping[str, '_TestSuite']):
        LOGGER.info("testsuites:\n" + "\n".join(testsuites) + "\n")
        summary = {}
        for name, testsuite in testsuites.items():
            LOGGER.info(f"running testsuite: {name}")
            try:
                summary[name] = self.run_testsuite(testsuite)
            except Exception as e:
                LOGGER.exception(f"testsuite {name} raise exception: {e}")
                summary[name] = None
            LOGGER.info(f"testsuite {name} completed\n")

        LOGGER.info(f"global summary:\n{_pretty_global_summary(summary)}")

    def __enter__(self):
        for tunnel, role_strings_list in self._tunnels:
            tunnel.start()
            for role_strings, address in zip(role_strings_list, tunnel.local_bind_addresses):
                client = _client_factory(client_type=self._client_type, address=address)
                for role_string in role_strings:
                    self._clients[role_string] = client
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for tunnel, _ in self._tunnels:
            try:
                tunnel.stop()
            except Exception as e:
                LOGGER.exception(e)

    def _get_client(self, role_string: str):
        return self._clients.get(role_string, self._default_client)


class _STATUS(Enum):
    ...


def _parse_address(address):
    host, port = address.split(":")
    port = int(port)
    return host, port


def _pretty_summary(summary: typing.List[typing.MutableMapping[str, str]]):
    table = prettytable.PrettyTable(field_names=["name", "job_id", "status"])
    for job_summary in summary:
        table.add_row([job_summary["name"], job_summary.get("job_id", "-"), job_summary["status"]])
    return table.get_string()


def _pretty_global_summary(summary: typing.MutableMapping[str, typing.List[typing.MutableMapping[str, str]]]):
    table = prettytable.PrettyTable(field_names=["testsuite", "name", "job_id", "status"])
    table.hrules = prettytable.ALL
    table.align["testsuite"] = "l"
    table.align["name"] = "l"
    table.max_width["testsuite"] = 30
    table.max_width["name"] = 20
    for name, testsuite_summary in summary.items():
        for job_summary in testsuite_summary:
            table.add_row([name, job_summary["name"], job_summary.get("job_id", "-"), job_summary["status"]])
    return table.get_string()


"""
client for submit job, upload data, query job
"""


def _client_factory(client_type, address: typing.Optional[typing.Union[str, typing.Tuple[str, int]]] = None):
    if isinstance(address, tuple):
        address = f"{address[0]}:{address[1]}"
    if client_type == "flowpy":
        return _FlowPYClient(address)
    if client_type == "rest":
        raise NotImplementedError(f"{client_type}")
        # return _RESTClient()
    raise NotImplementedError(f"{client_type}")


class _Client(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def upload_data(self, conf, verbose=0, drop=0) -> str:
        ...

    @abc.abstractmethod
    def query_job(self, job_id, role):
        ...

    @abc.abstractmethod
    def submit_job(self, conf, dsl) -> typing.Tuple[str, typing.MutableMapping[str, str]]:
        ...


class _RESTClient(_Client):

    def upload_data(self, conf, verbose=0, drop=0):
        ...

    def query_job(self, job_id, role):
        ...

    def submit_job(self, conf, dsl):
        ...


class _FlowPYClient(_Client):
    def __init__(self, address: typing.Optional[str] = None):

        from fate_flow.flowpy.client import FlowClient
        if address is not None:
            ip, port = _parse_address(address)
            self._client = FlowClient(ip=ip, port=port)
        else:
            self._client = FlowClient()

    def upload_data(self, conf, verbose=0, drop=0) -> str:
        with tempfile.NamedTemporaryFile("w") as f:
            json.dump(conf, f)
            f.flush()
            try:
                ret = self._client.data.upload(f.name, verbose=verbose, drop=drop)
            except Exception as e:
                LOGGER.exception(f"upload failed with conf:\n{pprint.pformat(conf)}")
                raise e
        if ret['retcode'] != 0:
            raise Exception(f"upload failed: {ret}\n"
                            f"conf: {pprint.pformat(conf)}")
        return ret["jobId"]

    def submit_job(self, conf, dsl) -> typing.Tuple[str, typing.MutableMapping[str, str]]:
        with tempfile.NamedTemporaryFile("w") as conf_file:
            with tempfile.NamedTemporaryFile("w") as dsl_file:
                json.dump(conf, conf_file)
                json.dump(dsl, dsl_file)
                conf_file.flush()
                dsl_file.flush()
                try:
                    ret = self._client.job.submit(conf_path=conf_file.name, dsl_path=dsl_file.name)
                except Exception as e:
                    LOGGER.exception(f"submit job failed with "
                                     f"conf:\n{pprint.pformat(conf)}\n"
                                     f"dsl:\n{pprint.pformat(dsl)}")
                    raise e
        if ret['retcode'] != 0:
            raise Exception(f"job submit fail: {ret}\n"
                            f"conf:\n{pprint.pformat(conf)}\n"
                            f"dsl:\n{pprint.pformat(dsl)}")
        return ret["jobId"], ret["data"]["model_info"]

    def query_job(self, job_id, role):
        while True:
            ret = self._client.job.query(job_id, role=role)
            if ret['retcode'] != 0:
                raise Exception(f"query job {job_id} fail: {ret}")
            status = ret['data'][0]["f_status"]
            if status in ["success", "failed", "canceled"]:
                return status
            time.sleep(1)


"""
data class from testsuite.json
"""


class _TestSuiteData(object):
    def __init__(self, file: str, head: int, partition: int, table_name: str, namespace: str, role_str: str):
        self.file = file
        self.head = head
        self.partition = partition
        self.table_name = table_name
        self.namespace = namespace
        self.role_str = role_str

    @classmethod
    def load(cls, config):
        kwargs = {}
        for k, v in cls.__init__.__annotations__.items():
            if k == "role":
                kwargs["role_str"] = v if v != "guest" else "guest_0"
            kwargs[k] = config.get(k)
        return _TestSuiteData(**kwargs)

    def dumps(self, fp, **kwargs):
        json.dump(self.as_dict(**kwargs), fp)

    def as_dict(self, **kwargs):
        d = {}
        for k, v in self.__annotations__.items():
            if k == "role":  # skip role
                continue
            d[k] = getattr(self, k)
        if kwargs:
            d.update(kwargs)
        return d


class _TestSuiteTask(object):
    def __init__(self, name: str, conf: dict,
                 dsl: typing.Optional[dict] = None,
                 deps: typing.Optional[str] = None):
        self.name = name
        self.conf = conf
        self.dsl = dsl
        self.deps = deps

    @classmethod
    def load(cls, name, config, base: Path, hook=None):
        kwargs = dict(name=name)
        with base.joinpath(config.get("conf")).resolve().open("r") as f:
            conf = json.load(f, object_hook=hook)

        dsl = config.get("dsl", None)
        if dsl is not None:
            with base.joinpath(dsl).resolve().open("r") as f:
                dsl = json.load(f)

        kwargs["conf"] = conf
        kwargs["dsl"] = dsl
        kwargs["deps"] = config.get("deps", None)
        return _TestSuiteTask(**kwargs)

    def update_job_parameters(self, **kwargs):
        self.conf["job_parameters"].update(**kwargs)

    def update_deps(self, deps_info):
        if self.deps is None:
            return
        if self.deps not in deps_info:
            raise Exception(f"task {self.name} depends on task {self.deps}")
        kwargs = deps_info[self.deps]
        self.conf["job_parameters"].update(**kwargs)

    def update_roles(self, role_to_parties):
        local_party_id = None
        for role, parties in role_to_parties.items():
            if role not in self.conf["role"]:
                continue
            num_parties = len(self.conf["role"][role])
            if role in self.conf["role"]:
                if num_parties > len(parties):
                    raise Exception(f"require {len(self.conf['role'][role])}, {len(parties)} provided: {parties}")
                self.conf["role"][role] = parties[:num_parties]
                if role == "guest":
                    local_party_id = parties[0]
        if local_party_id is None:
            raise Exception(f"no guest party provided")
        self.conf["initiator"] = {"party_id": local_party_id, "role": "guest"}


class _TestSuite(object):

    def __init__(self, data: typing.List[_TestSuiteData], task: typing.List[_TestSuiteTask]):
        self.data = data
        self.task = task

    @classmethod
    def load(cls, path: Path, hook):
        with path.open("r") as f:
            testsuite_config = json.load(f, object_hook=hook)
        data = [_TestSuiteData.load(d) for d in testsuite_config.get("data")]
        task = [_TestSuiteTask.load(name, config, path.parent, hook=hook) for name, config in
                testsuite_config.get("tasks").items()]
        return _TestSuite(data, task).reorder_task()

    def reorder_task(self):
        inplace = []
        delay = []
        deps = set()
        for t in self.task:
            if t.deps is not None and t.deps not in deps:
                delay.append(t)
            else:
                inplace.append(t)
                deps.add(t.name)
        self.task = inplace + delay
        return self

    def __str__(self):
        return f"{pprint.pformat(self.data)}\n{pprint.pformat(self.task)}"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    main()
