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
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Tuple

# noinspection PyPackageRequirements
import grpc

from arch.api.proto import federation_pb2, federation_pb2_grpc
from arch.api.transfer import Cleaner
from arch.api.transfer import Party, Federation
from arch.api.utils import eggroll_serdes
from arch.api.utils.log_utils import getLogger
from arch.api.utils.splitable import split_remote, is_split_head, make_fragment_tag, get_num_split, split_get
# noinspection PyProtectedMember
from eggroll.api.cluster.eggroll import _DTable, _EggRoll
from eggroll.api.proto import basic_meta_pb2, storage_basic_pb2

_ser_des = eggroll_serdes.PickleSerdes

OBJECT_STORAGE_NAME = "__federation__"

ERROR_STATES = [federation_pb2.CANCELLED, federation_pb2.ERROR]

LOGGER = getLogger()


async def _async_receive(stub, transfer_meta):
    resp_meta = stub.recv(transfer_meta)
    while resp_meta.transferStatus != federation_pb2.COMPLETE:
        if resp_meta.transferStatus in ERROR_STATES:
            raise IOError(
                "receive terminated, state: {}".format(federation_pb2.TransferStatus.Name(resp_meta.transferStatus)))
        resp_meta = stub.checkStatusNow(resp_meta)
        await asyncio.sleep(1)
    return resp_meta


def _thread_receive(receive_func, check_func, transfer_meta):
    resp_meta = receive_func(transfer_meta)
    while resp_meta.transferStatus != federation_pb2.COMPLETE:
        if resp_meta.transferStatus in ERROR_STATES:
            raise IOError(
                "receive terminated, state: {}".format(federation_pb2.TransferStatus.Name(resp_meta.transferStatus)))
        resp_meta = check_func(resp_meta)
    return resp_meta


class FederationRuntime(Federation):

    def __init__(self, session_id, runtime_conf, host, port):
        super().__init__(session_id, runtime_conf)

        channel = grpc.insecure_channel(target=f"{host}:{port}",
                                        options=[('grpc.max_send_message_length', -1),
                                                 ('grpc.max_receive_message_length', -1)])
        self._stub = federation_pb2_grpc.TransferSubmitServiceStub(channel)
        self.__pool = ThreadPoolExecutor()

    def remote(self, obj, name: str, tag: str, parties: Union[Party, list]) -> Cleaner:
        if isinstance(parties, Party):
            parties = [parties]
        self._remote_side_auth(name=name, parties=parties)

        # remote
        cleaner = Cleaner()
        for party in parties:
            tagged_key = self._make_tagged_key(name, tag, party)

            # remote dtable
            if isinstance(obj, _DTable):
                obj.set_gc_disable()
                # noinspection PyProtectedMember
                storage = storage_basic_pb2.StorageLocator(type=obj._type,
                                                           namespace=obj._namespace,
                                                           name=obj._name,
                                                           fragment=obj._partitions)
                data_desc = federation_pb2.TransferDataDesc(transferDataType=federation_pb2.DTABLE,
                                                            storageLocator=storage,
                                                            taggedVariableName=_ser_des.serialize(tagged_key))
                self._send(name, tag, party, tagged_key, data_desc)
                cleaner.add_table(obj)

            # remote obj
            else:
                # remote obj or fragment header
                table_name = f"{OBJECT_STORAGE_NAME}.{self._role}-{self._party_id}-{party.role}-{party.party_id}"
                table = _EggRoll.get_instance().table(table_name, self._session_id)
                value = split_remote(obj)
                table.put(tagged_key, value[0])
                self._send_obj(name, tag, party, tagged_key)
                cleaner.add_obj(table, tagged_key)

                # remote fragments
                if is_split_head(value[0]):
                    for k, v in value[1]:
                        fragment_tag = make_fragment_tag(tag, k)
                        fragment_tagged_key = self._make_tagged_key(name, fragment_tag, party)
                        table.put(fragment_tagged_key, v)
                        self._send_obj(name, fragment_tag, party, fragment_tagged_key)
                        cleaner.add_obj(table, fragment_tagged_key)
        return cleaner

    def get(self, name: str, tag: str, parties: Union[Party, list]) -> Tuple[list, Cleaner]:
        if isinstance(parties, Party):
            parties = [parties]
        self._get_side_auth(name=name, parties=parties)

        # async get
        metas = []
        for node in parties:
            metas.append((node, self._receive_meta(name, tag, node)))

        # block here, todo: any improvements?
        temps = [(node, meta()) for node, meta in metas]

        rtn = []
        cleaner = Cleaner()
        for node, recv_meta in temps:
            desc = recv_meta.dataDesc
            _persistent = desc.storageLocator.type != storage_basic_pb2.IN_MEMORY
            dest_table = _EggRoll.get_instance().table(name=desc.storageLocator.name,
                                                       namespace=desc.storageLocator.namespace,
                                                       persistent=_persistent)
            received = self.receive(recv_meta)
            if isinstance(received[0], _DTable):  # dtable
                value = received[0]
                cleaner.add_table(dest_table)
            elif is_split_head(received[0]):  # split object
                num_split = get_num_split(received[0])
                __tagged_key = received[1]
                fragment_metas = []
                for i in range(num_split):
                    fragment_metas.append(self._receive_meta(name, make_fragment_tag(tag, i), node))
                    fragment_key = make_fragment_tag(__tagged_key, i)
                    cleaner.add_obj(dest_table, fragment_key)
                fragment_metas = [meta() for meta in fragment_metas]
                fragments = [self.receive(meta) for meta in fragment_metas]
                value = split_get(fragments)
            else:  # object
                __tagged_key = received[1]
                cleaner.add_obj(dest_table, __tagged_key)
                value = received[0]
                LOGGER.debug("[GET] Got remote object {}".format(__tagged_key))
            rtn.append(value)
        return rtn, cleaner

    """internal utils"""

    def _make_tagged_key(self, name, tag, node):
        # warming: tag should be the last one
        return f"{self._session_id}-{node.role}-{node.party_id}-{name}-{tag}"

    def _send_obj(self, name, tag, node, tagged_key):

        storage = storage_basic_pb2.StorageLocator(type=storage_basic_pb2.LMDB,
                                                   namespace=self._session_id,
                                                   name=name)
        data_desc = federation_pb2.TransferDataDesc(transferDataType=federation_pb2.OBJECT,
                                                    storageLocator=storage,
                                                    taggedVariableName=_ser_des.serialize(tagged_key))
        self._send(name, tag, node, tagged_key, data_desc)

    def _send(self, name, tag, node, tagged_key, data_desc):
        LOGGER.debug(f"[REMOTE] Sending {tagged_key}")
        src = federation_pb2.Party(partyId=f"{self._party_id}", name=self._role)
        dst = federation_pb2.Party(partyId=f"{node.party_id}", name=node.role)
        job = basic_meta_pb2.Job(jobId=self._session_id, name=name)
        meta = federation_pb2.TransferMeta(
            job=job, tag=tag, src=src, dst=dst, dataDesc=data_desc, type=federation_pb2.SEND)
        self._stub.send(meta)
        LOGGER.debug(f"[REMOTE] Sent {tagged_key}")

    def _receive_meta(self, name, tag, node: Party):
        src = federation_pb2.Party(partyId=f"{node.party_id}", name=node.role)
        dst = federation_pb2.Party(partyId=f"{self._party_id}", name=self._role)
        trans_meta = federation_pb2.TransferMeta(job=basic_meta_pb2.Job(jobId=self._session_id, name=name),
                                                 tag=tag,
                                                 src=src,
                                                 dst=dst,
                                                 type=federation_pb2.RECV)

        return self.__pool.submit(_thread_receive, self._stub.recv, self._stub.checkStatus, trans_meta)

    @staticmethod
    def receive(recv_meta):
        desc = recv_meta.dataDesc
        _persistent = desc.storageLocator.type != storage_basic_pb2.IN_MEMORY
        dest_table = _EggRoll.get_instance().table(name=desc.storageLocator.name,
                                                   namespace=desc.storageLocator.namespace,
                                                   persistent=_persistent)
        if recv_meta.dataDesc.transferDataType == federation_pb2.OBJECT:
            __tagged_key = _ser_des.deserialize(desc.taggedVariableName)
            return dest_table.get(__tagged_key), __tagged_key
        else:
            src = recv_meta.src
            dst = recv_meta.dst
            LOGGER.debug(
                f"[GET] Got remote table {dest_table} from {src.name} {src.partyId} to {dst.name} {dst.partyId}")
            return dest_table,
