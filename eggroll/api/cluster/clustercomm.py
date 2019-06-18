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

import asyncio
import concurrent

import grpc

from eggroll.api.cluster.eggroll import _DTable, _EggRoll
from eggroll.api.proto import basic_meta_pb2, clustercomm_pb2, clustercomm_pb2_grpc, storage_basic_pb2
from eggroll.api.utils import file_utils, eggroll_serdes
from eggroll.api.utils.log_utils import getLogger

_serdes = eggroll_serdes.PickleSerdes

OBJECT_STORAGE_NAME = "__clustercomm__"

CONF_KEY_TARGET = "clustercomm"
CONF_KEY_LOCAL = "local"
CONF_KEY_SERVER = "servers"

ERROR_STATES = [clustercomm_pb2.CANCELLED, clustercomm_pb2.ERROR]


async def _async_receive(stub, transfer_meta):
    LOGGER.debug("start receiving {}".format(transfer_meta))
    resp_meta = stub.recv(transfer_meta)
    while resp_meta.transferStatus != clustercomm_pb2.COMPLETE:
        if resp_meta.transferStatus in ERROR_STATES:
            raise IOError(
                "receive terminated, state: {}".format(clustercomm_pb2.TransferStatus.Name(resp_meta.transferStatus)))
        resp_meta = stub.checkStatusNow(resp_meta)
        await asyncio.sleep(1)
    LOGGER.info("finish receiving {}".format(resp_meta))
    return resp_meta


def _thread_receive(receive_func, check_func, transfer_meta):
    LOGGER.debug("start receiving {}".format(transfer_meta))
    resp_meta = receive_func(transfer_meta)
    while resp_meta.transferStatus != clustercomm_pb2.COMPLETE:
        if resp_meta.transferStatus in ERROR_STATES:
            raise IOError(
                "receive terminated, state: {}".format(clustercomm_pb2.TransferStatus.Name(resp_meta.transferStatus)))
        resp_meta = check_func(resp_meta)
    LOGGER.info("finish receiving {}".format(resp_meta))
    return resp_meta


def init(job_id, runtime_conf, server_conf_path):
    global LOGGER
    LOGGER = getLogger()
    server_conf = file_utils.load_json_conf(server_conf_path)
    if CONF_KEY_SERVER not in server_conf:
        raise EnvironmentError("server_conf should contain key {}".format(CONF_KEY_SERVER))
    if CONF_KEY_TARGET not in server_conf.get(CONF_KEY_SERVER):
        raise EnvironmentError(
            "The {} should be a json file containing key: {}".format(server_conf_path, CONF_KEY_TARGET))
    _host = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_TARGET).get("host")
    _port = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_TARGET).get("port")
    if CONF_KEY_LOCAL not in runtime_conf:
        raise EnvironmentError("runtime_conf should be a dict containing key: {}".format(CONF_KEY_LOCAL))
    _party_id = runtime_conf.get(CONF_KEY_LOCAL).get("party_id")
    _role = runtime_conf.get(CONF_KEY_LOCAL).get("role")
    return ClusterCommRuntime(job_id, _party_id, _role, runtime_conf, _host, _port)


class ClusterCommRuntime(object):
    instance = None

    @staticmethod
    def __remote__object_key(*args):
        return "-".join(["{}".format(arg) for arg in args])

    @staticmethod
    def get_instance():
        if ClusterCommRuntime.instance is None:
            raise EnvironmentError("clustercomm should be initialized before use")
        return ClusterCommRuntime.instance

    def __init__(self, job_id, party_id, role, runtime_conf, host, port):
        self.trans_conf = file_utils.load_json_conf('conf/transfer_conf.json')
        self.job_id = job_id
        self.party_id = party_id
        self.role = role
        self.runtime_conf = runtime_conf
        self.channel = grpc.insecure_channel(
            target="{}:{}".format(host, port),
            options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
        self.stub = clustercomm_pb2_grpc.TransferSubmitServiceStub(self.channel)
        self.__pool = concurrent.futures.ThreadPoolExecutor()
        ClusterCommRuntime.instance = self

    def __get_locator(self, obj, name=None):
        if isinstance(obj, _DTable):
            return storage_basic_pb2.StorageLocator(type=obj._type, namespace=obj._namespace, name=obj._name,
                                                    fragment=obj._partitions)
        else:
            return storage_basic_pb2.StorageLocator(type=storage_basic_pb2.LMDB, namespace=self.job_id,
                                                    name=name)

    def __get_parties(self, role):
        return self.runtime_conf.get('role').get(role)

    def __check_authorization(self, name, is_send=True):
        algorithm, sub_name = name.split(".")
        auth_dict = self.trans_conf.get(algorithm)

        if auth_dict is None:
            raise ValueError("{} did not set in transfer_conf.json".format(algorithm))

        if auth_dict.get(sub_name) is None:
            raise ValueError("{} did not set under algorithm {} in transfer_conf.json".format(sub_name, algorithm))

        if is_send and auth_dict.get(sub_name).get('src') != self.role:
            raise ValueError("{} is not allow to send from {}".format(sub_name, self.role))
        elif not is_send and self.role not in auth_dict.get(sub_name).get('dst'):
            raise ValueError("{} is not allow to receive from {}".format(sub_name, self.role))
        return algorithm, sub_name

    def remote(self, obj, name: str, tag: str, role=None, idx=-1):
        algorithm, sub_name = self.__check_authorization(name)

        auth_dict = self.trans_conf.get(algorithm)

        src = clustercomm_pb2.Party(partyId="{}".format(self.party_id), name=self.role)

        if idx >= 0:
            if role is None:
                raise ValueError("{} cannot be None if idx specified".format(role))
            parties = {role: [self.__get_parties(role)[idx]]}
        elif role is not None:
            if role not in auth_dict.get(sub_name).get('dst'):
                raise ValueError("{} is not allowed to receive {}".format(role, name))
            parties = {role: self.__get_parties(role)}
        else:
            parties = {}
            for _role in auth_dict.get(sub_name).get('dst'):
                parties[_role] = self.__get_parties(_role)

        for _role, _partyIds in parties.items():
            for _partyId in _partyIds:
                _tagged_key = self.__remote__object_key(self.job_id, name, tag, self.role, self.party_id, _role,
                                                        _partyId)

                if isinstance(obj, _DTable):
                    '''
                    If it is a table, send the meta right away.
                    '''
                    desc = clustercomm_pb2.TransferDataDesc(transferDataType=clustercomm_pb2.DTABLE,
                                                           storageLocator=self.__get_locator(obj),
                                                           taggedVariableName=_serdes.serialize(_tagged_key))
                else:
                    '''
                    If it is a object, put the object in the table and send the table meta.
                    '''
                    _table = _EggRoll.get_instance().table(OBJECT_STORAGE_NAME, self.job_id)
                    _table.put(_tagged_key, obj)
                    storage_locator = self.__get_locator(_table)
                    desc = clustercomm_pb2.TransferDataDesc(transferDataType=clustercomm_pb2.OBJECT,
                                                           storageLocator=storage_locator,
                                                           taggedVariableName=_serdes.serialize(_tagged_key))

                LOGGER.debug("[REMOTE] Sending {}".format(_tagged_key))

                dst = clustercomm_pb2.Party(partyId="{}".format(_partyId), name=_role)
                job = basic_meta_pb2.Job(jobId=self.job_id, name=name)
                self.stub.send(clustercomm_pb2.TransferMeta(job=job, tag=tag, src=src, dst=dst, dataDesc=desc,
                                                           type=clustercomm_pb2.SEND))
                LOGGER.debug("[REMOTE] Sent {}".format(_tagged_key))

    def get(self, name, tag, idx=-1):
        algorithm, sub_name = self.__check_authorization(name, is_send=False)

        auth_dict = self.trans_conf.get(algorithm)

        src_role = auth_dict.get(sub_name).get('src')

        src_party_ids = self.__get_parties(src_role)

        if 0 <= idx < len(src_party_ids):
            # idx is specified, return the remote object
            party_ids = [src_party_ids[idx]]
        else:
            # idx is not valid, return remote object list
            party_ids = src_party_ids

        job = basic_meta_pb2.Job(jobId=self.job_id, name=name)

        LOGGER.debug(
            "[GET] {} {} getting remote object {} from {} {}".format(self.role, self.party_id, tag, src_role,
                                                                     party_ids))

        # loop = asyncio.get_event_loop()
        # tasks = []
        results = []
        for party_id in party_ids:
            src = clustercomm_pb2.Party(partyId="{}".format(party_id), name=src_role)
            dst = clustercomm_pb2.Party(partyId="{}".format(self.party_id), name=self.role)
            trans_meta = clustercomm_pb2.TransferMeta(job=job, tag=tag, src=src, dst=dst,
                                                     type=clustercomm_pb2.RECV)
            # tasks.append(_receive(self.stub, trans_meta))
            results.append(self.__pool.submit(_thread_receive, self.stub.recv, self.stub.checkStatus, trans_meta))
        # results = loop.run_until_complete(asyncio.gather(*tasks))
        # loop.close()
        results = [r.result() for r in results]
        rtn = []
        for recv_meta in results:
            desc = recv_meta.dataDesc
            _persistent = desc.storageLocator.type != storage_basic_pb2.IN_MEMORY
            dest_table = _EggRoll.get_instance().table(name=desc.storageLocator.name,
                                                       namespace=desc.storageLocator.namespace,
                                                       persistent=_persistent)
            if recv_meta.dataDesc.transferDataType == clustercomm_pb2.OBJECT:
                __tagged_key = _serdes.deserialize(desc.taggedVariableName)
                rtn.append(dest_table.get(__tagged_key))
                LOGGER.debug("[GET] Got remote object {}".format(__tagged_key))
            else:
                rtn.append(dest_table)
                src = recv_meta.src
                dst = recv_meta.dst
                LOGGER.debug(
                    "[GET] Got remote table {} from {} {} to {} {}".format(dest_table, src.name, src.partyId, dst.name,
                                                                           dst.partyId))
        if 0 <= idx < len(src_party_ids):
            return rtn[0]
        return rtn
