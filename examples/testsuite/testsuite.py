import abc
import argparse
import json
import pprint
import socket
import sys
import tempfile
import time
import traceback
import typing
from pathlib import Path

import loguru
import prettytable
import requests
import sshtunnel
import yaml
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor

LOGGER = loguru.logger

_HEAD = """\

                    ███████╗ █████╗ ████████╗███████╗                      
                    ██╔════╝██╔══██╗╚══██╔══╝██╔════╝                      
                    █████╗  ███████║   ██║   █████╗                        
                    ██╔══╝  ██╔══██║   ██║   ██╔══╝                        
                    ██║     ██║  ██║   ██║   ███████╗                      
                    ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚══════╝                      
                                                                           
████████╗███████╗███████╗████████╗    ███████╗██╗   ██╗██╗████████╗███████╗
╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██╔════╝██║   ██║██║╚══██╔══╝██╔════╝
   ██║   █████╗  ███████╗   ██║       ███████╗██║   ██║██║   ██║   █████╗  
   ██║   ██╔══╝  ╚════██║   ██║       ╚════██║██║   ██║██║   ██║   ██╔══╝  
   ██║   ███████╗███████║   ██║       ███████║╚██████╔╝██║   ██║   ███████╗
   ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚══════╝ ╚═════╝ ╚═╝   ╚═╝   ╚══════╝
                                                                    
"""

_TAIL = """\

    ██╗  ██╗ █████╗ ██╗   ██╗███████╗    ███████╗██╗   ██╗███╗   ██╗
    ██║  ██║██╔══██╗██║   ██║██╔════╝    ██╔════╝██║   ██║████╗  ██║
    ███████║███████║██║   ██║█████╗      █████╗  ██║   ██║██╔██╗ ██║
    ██╔══██║██╔══██║╚██╗ ██╔╝██╔══╝      ██╔══╝  ██║   ██║██║╚██╗██║
    ██║  ██║██║  ██║ ╚████╔╝ ███████╗    ██║     ╚██████╔╝██║ ╚████║
    ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝    ╚═╝      ╚═════╝ ╚═╝  ╚═══╝
                                                   
"""


def _show_head():
    print(_HEAD)


def _show_tail():
    print(_TAIL)


def main():
    _show_head()
    parser = argparse.ArgumentParser("TESTSUITE RUNNER")
    parser.add_argument("path", help="path to search xxx_testsuite.json")
    parser.add_argument("-drop", default=1, type=int, choices=[0, 1],
                        help="whether to drop the table if table already exists, 1 for yes and 0 for no")
    parser.add_argument("-config", default="./config.yaml", type=str,
                        help="config file")
    parser.add_argument("-name", default=f'testsuite-{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    parser.add_argument("-work_mode", choices=[0, 1], type=int)
    parser.add_argument("-client", choices=["flowpy", "rest"], type=str)
    parser.add_argument("-replace", type=str)
    parser.add_argument("--replace_config", type=str)
    parser.add_argument("-exclude", nargs="+", type=str)
    parser.add_argument("-skip_data", default=False, type=bool)
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
    elif args.replace_config:
        replace_config_path = Path(args.replace_config).resolve()
        if not replace_config_path.exists():
            raise ValueError(f"{replace_config_path} not exists")

        with replace_config_path.open("r") as f:
            if args.replace_config.endswith(".yaml"):
                s = yaml.load(f)
            else:
                s = json.load(f)
        hook = _replace_hook(s)

    with Clients(config_path=Path(args.config), drop=args.drop, **config_overwrite) as clients:
        paths = _find_testsuite_files(path)

        # exclude
        if args.exclude is not None:
            exclude_paths = set()
            for p in args.exclude:
                exclude_paths.update(_find_testsuite_files(Path(p).resolve()))
            paths = [p for p in paths if p not in exclude_paths]
        testsuites = {path.__str__(): _TestSuite.load(path, hook=hook) for path in paths}
        _list_testsuites(testsuites)
        summaries = clients.run_testsuites(testsuites,
                                           summaries_base=Path(args.name).resolve(),
                                           skip_data=args.skip_data)

    with Path(args.name).joinpath("summaries").open("w") as f:
        f.write(summaries.pretty_summaries())
        f.write("\n")
        f.write("unsuccessful:\n")
        f.write(summaries.pretty_summaries(include_success=False))
    print(f"summaries:\n{summaries.pretty_summaries()}")
    print()
    print(f"unsuccessful summaries:\n{summaries.pretty_summaries(include_success=False)}")
    _show_tail()
    print(f"LOG: {args.name}")


def _find_testsuite_files(path):
    if path.is_file():
        if path.name.endswith("testsuite.json"):
            paths = [path]
        else:
            LOGGER.warning(f"{path} is file, but not end with `testsuite.json`, skip")
            paths = []
    else:
        paths = path.glob(f"**/*testsuite.json")
    return [p.resolve() for p in paths]


def _replace_hook(mapping: dict):
    def _hook(d):
        for k, v in mapping.items():
            if k in d:
                d[k] = v
        return d

    return _hook


def _add_logger(name):
    path = Path(name).joinpath("logs")
    if path.exists() and not path.is_dir():
        raise Exception(f"{name} exist, but is not a dir")
    if not path.exists():
        path.mkdir(parents=True)
    loguru.logger.remove()
    simple_log_format = '<green>[{time:HH:mm:ss}]</green><level>{message}</level>'
    log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | ' \
                 '<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    loguru.logger.add(sys.stderr, level="INFO", colorize=True, format=simple_log_format)
    loguru.logger.add(f"{path.joinpath('info.log')}", level="INFO", format=log_format)
    loguru.logger.add(f"{path.joinpath('debug.log')}", level="DEBUG", format=log_format)


class Clients(object):
    def __init__(self, config_path: Path, **kwargs):
        with config_path.open("r") as f:
            conf = yaml.load(f)
        if kwargs:
            conf.update(kwargs)

        LOGGER.debug(f"config: {pprint.pformat(conf)}")
        self._client_type = conf.get("client", "flowpy").lower()
        self._drop = conf.get("drop", 0)
        self._work_mode = int(conf.get("work_mode", "0"))
        if conf.get("data_base_dir") is not None:
            data_base_dir = Path(conf["data_base_dir"])
            if not data_base_dir.is_absolute():
                data_base_dir = config_path.parent.joinpath(data_base_dir)
            data_base_dir = data_base_dir.resolve()
        else:
            data_base_dir = None
        self._data_base_dir = data_base_dir

        self._role_to_parties = conf.get("parties")
        parties_to_role_string = self._parties_to_role_string(self._role_to_parties)
        self._tunnels = self._create_ssh_tunnels(conf.get("ssh_tunnel", []), parties_to_role_string)
        self._clients = self._local_clients(conf, self._client_type, parties_to_role_string, self._data_base_dir)

    def run_testsuite(self, testsuite: '_TestSuite', skip_data=False) -> '_Summary':
        num_data = len(testsuite.data)
        # upload data, raise exception if any exception occurs or data upload job failed.
        if not skip_data:
            for i, data in enumerate(testsuite.data):
                print()
                LOGGER.info(f"upload data(({i + 1}/{num_data})): role: {data.role_str}, file: {data.file}")
                client = self._get_client(data.role_str)

                # submit job
                job_id = client.upload_data(data.as_dict(work_mode=self._work_mode), drop=self._drop)
                LOGGER.opt(colors=True).info(f"job id: <green>{job_id}</green>")

                # check status
                try:
                    data_upload_checker = client.query_job(job_id=job_id, role="local")
                    while True:
                        next(data_upload_checker)
                except StopIteration as e:
                    status = e.value
                    if status == "success":
                        LOGGER.opt(colors=True).info(f"done: <green>{status}</green>")
                    else:
                        LOGGER.opt(colors=True).info(f"done: <red>{status}</red>")
                if status != "success":
                    raise Exception(f"failed to upload {i + 1}th data: {status}")

        # submit jobs, jobs's exception will logged then ignored
        num_task = len(testsuite.task)
        deps_info = {}
        summary = _Summary(testsuite.path)
        for i, task in enumerate(testsuite.task):
            task_summary = {"task": task}
            try:
                start = time.time()
                print()
                LOGGER.info(f"submit job({i + 1}/{num_task}): {task.name}")
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
                LOGGER.opt(colors=True).info(f"job_id: <green>{job_id}</green>")

                # check status, block until job is completed
                try:
                    job_submit_checker = client.query_job(job_id=job_id, role="guest")
                    while True:
                        progress = next(job_submit_checker)
                        LOGGER.info(_progress_bar(progress, 50))
                except StopIteration as e:
                    status = e.value

                task_summary["status"] = status
                if status == "success":
                    deps_info[task.name] = model_info
                    LOGGER.opt(colors=True).info(f"done: <green>{status}</green>")
                else:
                    LOGGER.opt(colors=True).info(f"done: <red>{status}</red>")
                LOGGER.info(f"takes {time.time() - start:.2f}s")
            except Exception as e:
                LOGGER.exception(f"task {task.name} error, {e}")
                task_summary["status"] = "exception"
                if "job_id" not in task_summary:
                    task_summary["job_id"] = _get_next_exception_id()
            summary.add_task_summary(**task_summary)
        return summary

    def run_testsuites(self, testsuites: typing.MutableMapping[str, '_TestSuite'],
                       summaries_base: typing.Optional[Path] = None,
                       skip_data=False):
        num_testsuites = len(testsuites)
        summaries = _Summaries()
        testsuite_index = 1
        for name, testsuite in testsuites.items():
            print()
            LOGGER.info(f"testsuite({testsuite_index}/{num_testsuites}):\n"
                        f"num_data={len(testsuite.data)}\n"
                        f"num_jobs={len(testsuite.task)}\n"
                        f"path={testsuite.path}")
            try:
                summary = self.run_testsuite(testsuite, skip_data)
                print()
                print(summary.pretty_summary())

                if summaries_base is not None:
                    summary.write(base=summaries_base)
                summaries.add_summary(summary)

            except Exception as e:
                LOGGER.exception(f"testsuite {name} raise exception: {e}")
                summaries.add_summary(_Summary(testsuite.path))
            LOGGER.info(f"testsuite done({testsuite_index}/{num_testsuites}): {name}\n")
            testsuite_index += 1
        return summaries

    def __enter__(self):
        return self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for tunnel, _ in self._tunnels:
            try:
                tunnel.stop()
            except Exception as e:
                LOGGER.exception(e)

    @staticmethod
    def _create_ssh_tunnels(ssh_tunnel_conf, parties_to_role_string) -> \
            typing.List[typing.Tuple[sshtunnel.SSHTunnelForwarder, typing.List[typing.MutableSet[str]]]]:

        tunnels = []
        for ssh_conf in ssh_tunnel_conf:
            ssh_address = ssh_conf.get("ssh_address")
            ssh_host, ssh_port = _parse_address(ssh_address)
            ssh_username = ssh_conf.get("ssh_username")
            ssh_password = ssh_conf.get("ssh_password")
            ssh_pkey = ssh_conf.get("ssh_priv_key")
            services = ssh_conf.get("services")

            role_strings = []
            remote_bind_addresses = []
            for service in services:
                role_string_set = set()
                for party in service.get("parties"):
                    role_string_set.update(parties_to_role_string[party])
                role_strings.append(role_string_set)
                remote_bind_addresses.append(_parse_address(service.get("address")))

            tunnel = sshtunnel.SSHTunnelForwarder(ssh_address_or_host=(ssh_host, ssh_port),
                                                  ssh_username=ssh_username,
                                                  ssh_password=ssh_password,
                                                  ssh_pkey=ssh_pkey,
                                                  remote_bind_addresses=remote_bind_addresses)
            tunnels.append((tunnel, role_strings))
        return tunnels

    @staticmethod
    def _local_clients(conf, client_type, parties_to_role_string, data_base_dir) -> \
            typing.MutableMapping[str, '_Client']:
        clients = {}
        for service in conf.get("local_services"):
            client = _client_factory(client_type, service["address"], data_base_dir)
            for party in service["parties"]:
                for role_str in parties_to_role_string[party]:
                    clients[role_str] = client
        return clients

    @staticmethod
    def _parties_to_role_string(role_to_parties):
        parties_to_role_string = {}
        for role, parties in role_to_parties.items():
            for i, party in enumerate(parties):
                if party not in parties_to_role_string:
                    parties_to_role_string[party] = set()
                parties_to_role_string[party].add(f"{role.lower()}_{i}")
        return parties_to_role_string

    def _open(self):
        for tunnel, role_strings_list in self._tunnels:
            tunnel.start()
            for role_strings, address in zip(role_strings_list, tunnel.local_bind_addresses):
                client = _client_factory(client_type=self._client_type, address=address,
                                         data_base_dir=self._data_base_dir)
                for role_string in role_strings:
                    self._clients[role_string] = client
        return self

    def _get_client(self, role_string: str):
        if role_string not in self._clients:
            raise Exception(f"{role_string} not found in {self._clients}")
        return self._clients.get(role_string)


_exception_id = 0


def _progress_bar(progress, max_size=50):
    finished = int(progress / 100.0 * max_size)
    unfinished = max_size - finished
    bar = "█" * finished + "░" * unfinished + f" {progress}%"
    return bar


def _get_next_exception_id():
    global _exception_id
    _exception_id += 1
    return f"exception_task_{_exception_id}"


def _parse_address(address):
    host, port = address.split(":")
    port = int(port)
    return host, port


def _list_testsuites(testsuites: typing.MutableMapping[str, '_TestSuite']):
    table = prettytable.PrettyTable(field_names=["testsuite", "num_data", "num_jobs"])
    table.hrules = prettytable.ALL
    table.align["testsuite"] = "l"
    table.max_width["testsuite"] = 40
    for testsuite in testsuites.values():
        table.add_row([testsuite.path, len(testsuite.data), len(testsuite.task)])
    pretty_str = table.get_string()
    print(pretty_str)
    LOGGER.debug(f"testsuite_info:\n{pretty_str}")


class _Summary(object):
    def __init__(self, path: Path):
        self.path = path
        self.tasks = []

    def add_task_summary(self, status, job_id, task):
        self.tasks.append((status, job_id, task))

    def pretty_summary(self):
        table = prettytable.PrettyTable(field_names=["name", "job_id", "status"])
        for status, job_id, task in self.tasks:
            table.add_row([task.name, job_id, status])
        return table.get_string()

    def write(self, base: Path):
        for status, job_id, task in self.tasks:
            task_log_path = base.joinpath(status).joinpath(job_id)
            task_log_path.mkdir(parents=True, exist_ok=True)
            with task_log_path.joinpath("conf").open("w") as f:
                json.dump(task.conf, f, indent=2)
            if task.dsl is not None:
                with task_log_path.joinpath("dsl").open("w") as f:
                    json.dump(task.dsl, f, indent=2)
        with base.joinpath("summary").open("a") as f:
            f.write(f"testsuite: {self.path}\n")
            f.write(self.pretty_summary())
            f.write("\n")


class _Summaries(object):
    def __init__(self):
        self._summaries: typing.List[_Summary] = []

    def add_summary(self, summary):
        self._summaries.append(summary)

    def pretty_summaries(self, include_success=True):
        table = prettytable.PrettyTable(field_names=["testsuite", "name", "job_id", "status"])
        table.hrules = prettytable.ALL
        table.align["testsuite"] = "l"
        table.align["name"] = "l"
        table.max_width["testsuite"] = 30
        table.max_width["name"] = 20
        for summary in self._summaries:
            for status, job_id, task in summary.tasks:
                if not include_success and status == "success":
                    continue
                table.add_row([summary.path.__str__(), task.name, job_id, status])
        return table.get_string()


"""
client for submit job, upload data, query job
"""


def _client_factory(client_type,
                    address: typing.Optional[typing.Union[str, typing.Tuple[str, int]]] = None,
                    data_base_dir: typing.Optional[Path] = None):
    if isinstance(address, tuple):
        address = f"{address[0]}:{address[1]}"
    if client_type == "flowpy":
        return _FlowPYClient(address, data_base_dir)
    if client_type == "rest":
        return _RESTClient(address, data_base_dir)
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

    def __init__(self, address: typing.Optional[str] = None, data_base_dir: typing.Optional[Path] = None):
        if address is None:
            address = f"{socket.gethostbyname(socket.gethostname())}:9380"
        self.address = address
        self.version = "v1"
        self._base = f"http://{self.address}/{self.version}/"
        self._http = requests.Session()
        self._data_base_dir = data_base_dir
        if self._data_base_dir is None:
            self._data_base_dir = Path(__file__).resolve().parent.parent.parent

    @LOGGER.catch
    def _post(self, url, **kwargs) -> dict:
        request_url = self._base + url
        try:
            response = self._http.request(method='post', url=request_url, **kwargs)
        except Exception as e:
            LOGGER.exception(e)
            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            response = {'retcode': 100, 'retmsg': str(e),
                        'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback_obj)}
            if 'Connection refused' in str(e):
                response['retmsg'] = 'Connection refused, Please check if the fate flow service is started'
                del response['traceback']
            return response

        try:
            if isinstance(response, requests.models.Response):
                response = response.json()
            else:
                try:
                    response = json.loads(response.content.decode('utf-8', 'ignore'), strict=False)
                except (TypeError, ValueError):
                    return response
        except json.decoder.JSONDecodeError:
            response = {'retcode': 100,
                        'retmsg': "Internal server error. Nothing in response. You may check out the configuration in "
                                  "'FATE/arch/conf/server_conf.json' and restart fate flow server."}
        return response

    def upload_data(self, conf, verbose=0, drop=0):
        conf['drop'] = drop if drop else 2
        conf['verbose'] = verbose
        path = Path(conf.get('file'))
        if not path.is_file():
            path = self._data_base_dir.joinpath(conf.get('file')).resolve()

        if not path.exists():
            raise Exception('The file is obtained from the fate flow client machine, but it does not exist, '
                            f'please check the path: {path}')

        with path.open("rb") as fp:
            data = MultipartEncoder(
                fields={'file': (path.name, fp, 'application/octet-stream')}
            )
            tag = [0]

            def read_callback(monitor):
                if conf.get('verbose') == 1:
                    sys.stdout.write(
                        "\r UPLOADING:{0}{1}".format("|" * (monitor.bytes_read * 100 // monitor.len),
                                                     '%.2f%%' % (monitor.bytes_read * 100 // monitor.len)))
                    sys.stdout.flush()
                    if monitor.bytes_read / monitor.len == 1:
                        tag[0] += 1
                        if tag[0] == 2:
                            sys.stdout.write('\n')

            data = MultipartEncoderMonitor(data, read_callback)
            response = self._post(url='data/upload', data=data, params=conf,
                                  headers={'Content-Type': data.content_type})

        if response['retcode'] != 0:
            raise Exception(f"upload failed: {response}\n"
                            f"conf: {pprint.pformat(conf)}")
        return response["jobId"]

    def query_job(self, job_id, role):
        data = {"local": {}, "job_id": str(job_id)}
        if role is not None:
            data["local"]["role"] = role

        pre_progress = -1
        while True:
            ret = self._post(url='job/query', json=data)
            if ret['retcode'] != 0:
                raise Exception(f"query job {job_id} fail: {ret}")
            status = ret['data'][0]["f_status"]
            if status in ["success", "failed", "canceled"]:
                return status
            progress = ret['data'][0]["f_progress"]
            if status == "running" and pre_progress != progress:
                yield progress
                pre_progress = progress
            time.sleep(0.5)

    def submit_job(self, conf, dsl):
        post_data = {
            'job_dsl': dsl,
            'job_runtime_conf': conf
        }
        response = self._post(url='job/submit', json=post_data)

        if response['retcode'] != 0:
            raise Exception(f"job submit fail: {response}\n"
                            f"conf:\n{pprint.pformat(conf)}\n"
                            f"dsl:\n{pprint.pformat(dsl)}")
        return response["jobId"], response["data"]["model_info"]


class _FlowPYClient(_Client):
    def __init__(self, address: typing.Optional[str] = None, data_base_dir: typing.Optional[Path] = None):

        from fate_flow.flowpy.client import FlowClient
        if address is not None:
            ip, port = _parse_address(address)
            self._client = FlowClient(ip=ip, port=port)
        else:
            self._client = FlowClient()

        self._data_base_dir = data_base_dir
        if self._data_base_dir is None:
            self._data_base_dir = Path(__file__).resolve().parent.parent.parent

    def upload_data(self, conf, verbose=0, drop=0) -> str:
        # change data base
        if not Path(conf["file"]).is_file():
            conf["file"] = self._data_base_dir.joinpath(conf["file"]).resolve()

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
        pre_progress = -1
        while True:
            ret = self._client.job.query(job_id, role=role)
            if ret['retcode'] != 0:
                raise Exception(f"query job {job_id} fail: {ret}")
            status = ret['data'][0]["f_status"]
            if status in ["success", "failed", "canceled"]:
                return status
            progress = ret['data'][0]["f_progress"]
            if status == "running" and pre_progress != progress:
                yield progress
                pre_progress = progress
            time.sleep(0.5)


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
        return _TestSuiteData(
            file=config.get("file"),
            head=config.get("head"),
            partition=config.get("partition"),
            table_name=config.get("table_name"),
            namespace=config.get("namespace"),
            role_str=config.get("role") if config.get("role") != "guest" else "guest_0"
        )

    def dumps(self, fp, **kwargs):
        json.dump(self.as_dict(**kwargs), fp)

    def as_dict(self, **kwargs):
        d = dict(file=self.file,
                 head=self.head,
                 partition=self.partition,
                 table_name=self.table_name,
                 namespace=self.namespace)
        if kwargs:
            d.update(kwargs)
        return d

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


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

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


class _TestSuite(object):

    def __init__(self, data: typing.List[_TestSuiteData], task: typing.List[_TestSuiteTask], path: Path):
        self.data = data
        self.task = task
        self.path = path

    @classmethod
    def load(cls, path: Path, hook):
        with path.open("r") as f:
            testsuite_config = json.load(f, object_hook=hook)
        data = [_TestSuiteData.load(d) for d in testsuite_config.get("data")]
        task = [_TestSuiteTask.load(name, config, path.parent, hook=hook) for name, config in
                testsuite_config.get("tasks").items()]
        return _TestSuite(data, task, path).reorder_task()

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
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    main()
