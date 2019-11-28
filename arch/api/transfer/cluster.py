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
from arch.api.proto.federation_pb2 import TransferMeta, TransferDataDesc
from arch.api.transfer import Cleaner
from arch.api.transfer import Party, Federation
from arch.api.utils import eggroll_serdes
from arch.api.utils.log_utils import getLogger
from arch.api.utils.splitable import maybe_split_object, is_split_head, split_get
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
        self._cache_remote_obj_storage_table = {}
        self._cache_get_obj_storage_table = {}

    def remote(self, obj, name: str, tag: str, parties: Union[Party, list]) -> Cleaner:
        if isinstance(parties, Party):
            parties = [parties]
        self._remote_side_auth(name=name, parties=parties)
        cleaner = Cleaner()

        # if obj is a dtable, remote it
        if isinstance(obj, _DTable):
            obj.set_gc_disable()
            for party in parties:
                self._send(transfer_type=federation_pb2.DTABLE, name=name, tag=tag, dst_party=party, cleaner=cleaner,
                           table=obj)
            return cleaner

        # if obj is object, put it in specified dtable, then remote the dtable
        first, fragments = maybe_split_object(obj)
        for party in parties:
            obj_table = self._remote_side_obj_storage_table(dst_party=party)

            # remote object or remote fragment header
            self._send(transfer_type=federation_pb2.OBJECT, name=name, tag=tag, dst_party=party, cleaner=cleaner,
                       table=obj_table, obj=first)

            # remote fragments
            for fragment_index, fragment in fragments:
                self._send(transfer_type=federation_pb2.OBJECT, name=name, tag=self._fragment_tag(tag, fragment_index),
                           dst_party=party, cleaner=cleaner, table=obj_table, obj=fragment)
        return cleaner

    def get(self, name: str, tag: str, parties: Union[Party, list]) -> Tuple[list, Cleaner]:
        if isinstance(parties, Party):
            parties = [parties]
        self._get_side_auth(name=name, parties=parties)
        cleaner = Cleaner()

        # async get
        meta_futures = [(party, self._receive_meta(name, tag, party)) for party in parties]
        # block here, todo: any improvements?
        received_metas = [(party, meta.result()) for party, meta in meta_futures]

        rtn = []
        for party, recv_meta in received_metas:
            desc = recv_meta.dataDesc
            if desc.transferDataType == federation_pb2.DTABLE:
                dest_table = self._get_table(recv_meta.dataDesc)
                src = recv_meta.src
                dst = recv_meta.dst
                LOGGER.debug(
                    f"[GET] Got remote table {dest_table} from {src.name} {src.partyId} to {dst.name} {dst.partyId}")
                cleaner.add_table(dest_table)
                rtn.append(dest_table)
            elif desc.transferDataType == federation_pb2.OBJECT:
                obj_table = self._get_side_obj_storage_table(src_party=party)
                __tagged_key = _ser_des.deserialize(desc.taggedVariableName)
                obj = obj_table.get(__tagged_key)

                if not is_split_head(obj):
                    cleaner.add_obj(obj_table, __tagged_key)
                    rtn.append(obj)
                else:
                    num_split = obj.num_split()
                    fragments = []
                    fragment_meta_futures = [self._receive_meta(name, self._fragment_tag(tag, i), party) for i in
                                             range(num_split)]
                    fragment_metas = [meta.result() for meta in fragment_meta_futures]
                    for meta in fragment_metas:
                        __fragment_tagged_key = _ser_des.deserialize(meta.dataDesc.taggedVariableName)
                        fragments.append(obj_table.get(__fragment_tagged_key))
                        cleaner.add_obj(obj_table, __fragment_tagged_key)
                    rtn.append(split_get(fragments))
            else:
                raise IOError(f"unknown transfer type: {recv_meta.dataDesc.transferDataType}")
        return rtn, cleaner

    """internal utils"""

    @staticmethod
    def _fragment_tag(tag, idx):
        return f"{tag}.__frag__{idx}"

    def _create_storage_table(self, src_party, dst_party):
        name = f"{OBJECT_STORAGE_NAME}.{src_party.role}-{src_party.party_id}-{dst_party.role}-{dst_party.party_id}"
        return _EggRoll.get_instance().table(name=name, namespace=self._session_id)

    def _remote_side_obj_storage_table(self, dst_party: Party):
        if dst_party in self._cache_remote_obj_storage_table:
            return self._cache_remote_obj_storage_table[dst_party]

        table = self._create_storage_table(self.local_party, dst_party)
        self._cache_remote_obj_storage_table[dst_party] = table
        return table

    def _get_side_obj_storage_table(self, src_party: Party):
        if src_party in self._cache_get_obj_storage_table:
            return self._cache_get_obj_storage_table[src_party]

        table = self._create_storage_table(src_party, self.local_party)
        self._cache_get_obj_storage_table[src_party] = table
        return table

    @staticmethod
    def _get_table(transfer_data_desc):
        name = transfer_data_desc.storageLocator.name,
        namespace = transfer_data_desc.storageLocator.namespace,
        persistent = transfer_data_desc.storageLocator.type != storage_basic_pb2.IN_MEMORY
        return _EggRoll.get_instance().table(name=name, namespace=namespace, persistent=persistent)

    def _send(self, transfer_type, name: str, tag: str, dst_party: Party, cleaner: Cleaner, table: _DTable, obj=None):
        tagged_key = f"{name}-{tag}"
        LOGGER.debug(f"[REMOTE] sending: type={transfer_type}, table={table}, tagged_key={tagged_key}")
        if transfer_type == federation_pb2.OBJECT:
            table.put(tagged_key, obj)
            cleaner.add_obj(table, tagged_key)
        else:
            cleaner.add_table(table)
        data_desc = TransferDataDesc(transferDataType=transfer_type,
                                     storageLocator=self.get_storage_locator(table),
                                     taggedVariableName=_ser_des.serialize(tagged_key))
        job = basic_meta_pb2.Job(jobId=self._session_id, name=name)
        transfer_meta = TransferMeta(job=job, tag=tag, src=self.local_party.to_pb(), dst=dst_party.to_pb(),
                                     dataDesc=data_desc, type=federation_pb2.SEND)
        self._stub.send(transfer_meta)
        LOGGER.debug(f"[REMOTE] done: type={transfer_type}, table={table}, tagged_key={tagged_key}")

    def _receive_meta(self, name, tag, party: Party):
        job = basic_meta_pb2.Job(jobId=self._session_id, name=name)
        transfer_meta = TransferMeta(job=job, tag=tag, src=party.to_pb(), dst=self.local_party.to_pb(),
                                     type=federation_pb2.RECV)
        return self.__pool.submit(_thread_receive, self._stub.recv, self._stub.checkStatus, transfer_meta)

    # noinspection PyProtectedMember
    @staticmethod
    def get_storage_locator(table):
        return storage_basic_pb2.StorageLocator(type=table._type,
                                                namespace=table._namespace,
                                                name=table._name,
                                                fragment=table._partitions)
