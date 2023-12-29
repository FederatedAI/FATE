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

import concurrent.futures
import logging
import os
import signal
import typing
from typing import List

from eggroll.computing import RollPair
from eggroll.federation import RollSiteContext
from fate.arch.computing.backends.eggroll import Table
from fate.arch.federation.api import Federation, PartyMeta, TableMeta

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from fate.arch.computing.backends.eggroll import CSession


class EggrollFederation(Federation):
    def __init__(
        self,
        computing_session: "CSession",
        federation_session_id,
        party: PartyMeta,
        parties: List[PartyMeta],
        proxy_endpoint,
    ):
        super().__init__(federation_session_id, party, parties)
        proxy_endpoint_host, proxy_endpoint_port = proxy_endpoint.split(":")
        self._rp_ctx = computing_session.get_rpc()
        self._rsc = RollSiteContext(
            federation_session_id,
            rp_ctx=self._rp_ctx,
            party=party,
            proxy_endpoint_host=proxy_endpoint_host.strip(),
            proxy_endpoint_port=int(proxy_endpoint_port.strip()),
        )

    def _pull_table(
        self, name: str, tag: str, parties: List[PartyMeta], table_metas: List[TableMeta]
    ) -> List["Table"]:
        rs = self._rsc.load(name=name, tag=tag)
        future_map = dict(
            zip(
                rs.pull(parties=parties),
                parties,
            )
        )
        rtn = {}
        for future in concurrent.futures.as_completed(future_map):
            party = future_map[future]
            v = future.result()
            assert isinstance(v, RollPair), f"pull table got {type(v)}"
            rtn[party] = Table(v)
        return [rtn[party] for party in parties]

    def _pull_bytes(self, name: str, tag: str, parties: List[PartyMeta]):
        rs = self._rsc.load(name=name, tag=tag)
        future_map = dict(
            zip(
                rs.pull(parties=parties),
                parties,
            )
        )
        rtn = {}
        for future in concurrent.futures.as_completed(future_map):
            party = future_map[future]
            v = future.result()
            rtn[party] = v

        return [rtn[party] for party in parties]

    def _push_table(self, table: Table, name: str, tag: str, parties: List[PartyMeta]):
        rs = self._rsc.load(name=name, tag=tag)
        futures = rs.push_rp(table._rp, parties=parties)
        # done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        # for future in done:
        #     future.result()

    def _push_bytes(self, v: bytes, name: str, tag: str, parties: List[PartyMeta]):
        rs = self._rsc.load(name=name, tag=tag)
        futures = rs.push_bytes(v, parties=parties)
        # done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
        # for future in done:
        #     future.result()

    def _destroy(self):
        self._rp_ctx.cleanup(name="*", namespace=self._session_id)


def _push_with_exception_handle(rsc, v, name: str, tag: str, parties: List[PartyMeta]):
    def _remote_exception_re_raise(f, p: PartyMeta):
        try:
            f.result()
            logger.debug(f"[federation.eggroll.remote.{name}.{tag}]future to remote to party: {p} done")
        except Exception as e:
            pid = os.getpid()
            logger.exception(
                f"[federation.eggroll.remote.{name}.{tag}]future to remote to party: {p} fail,"
                f" terminating process(pid={pid})"
            )
            import traceback

            print(
                f"federation.eggroll.remote.{name}.{tag} future to remote to party: {p} fail,"
                f" terminating process {pid}, traceback: {traceback.format_exc()}"
            )
            os.kill(pid, signal.SIGTERM)
            raise e

    def _get_call_back_func(p: PartyMeta):
        def _callback(f):
            return _remote_exception_re_raise(f, p)

        return _callback

    rs = rsc.load(name=name, tag=tag)
    futures = rs.push(obj=v, parties=parties)
    for party, future in zip(parties, futures):
        future.add_done_callback(_get_call_back_func(party))

    add_remote_futures(futures)
    return rs


_remote_futures = set()


def _clear_callback(future):
    logger.debug("future `{future}` done, remove")
    _remote_futures.remove(future)


def add_remote_futures(fs: typing.List[concurrent.futures.Future]):
    for f in fs:
        f.add_done_callback(_clear_callback)
        _remote_futures.add(f)


def wait_all_remote_done(timeout=None):
    concurrent.futures.wait(_remote_futures, timeout=timeout, return_when=concurrent.futures.FIRST_EXCEPTION)
