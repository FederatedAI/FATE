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

from eggroll.roll_pair.roll_pair import RollPair
from eggroll.roll_site.roll_site import RollSiteContext
from fate.arch.abc import FederationEngine, PartyMeta

from ...computing.eggroll import Table
from .._gc import GarbageCollector

LOGGER = logging.getLogger(__name__)


class EggrollFederation(FederationEngine):
    def __init__(
        self,
        rp_ctx,
        rs_session_id,
        party: PartyMeta,
        parties: List[PartyMeta],
        proxy_endpoint,
    ):
        LOGGER.debug(
            f"[federation.eggroll]init federation: "
            f"rp_session_id={rp_ctx.session_id}, rs_session_id={rs_session_id}, "
            f"party={party}, proxy_endpoint={proxy_endpoint}"
        )

        options = {
            "self_role": party[0],
            "self_party_id": party[1],
            "proxy_endpoint": proxy_endpoint,
        }
        self._session_id = rs_session_id
        self._rp_ctx = rp_ctx
        self._get_history = set()
        self._remote_history = set()

        self.get_gc: GarbageCollector = GarbageCollector()
        self.remote_gc: GarbageCollector = GarbageCollector()
        self.local_party = party
        self.parties = parties
        self._rsc = RollSiteContext(rs_session_id, rp_ctx=rp_ctx, options=options)
        LOGGER.debug(f"[federation.eggroll]init federation context done")

    @property
    def session_id(self) -> str:
        return self._session_id

    def pull(self, name: str, tag: str, parties: List[PartyMeta]):
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
            log_str = f"federation.eggroll.get.{name}.{tag}"
            if v is None:
                raise ValueError(f"[{log_str}]get `None` from {party}")
            if (name, tag, party) in self._get_history:
                raise ValueError(f"[{log_str}]get from {party} with duplicate tag")
            self._get_history.add((name, tag, party))
            # got a roll pair
            if isinstance(v, RollPair):
                LOGGER.debug(
                    f"[{log_str}] got "
                    f"RollPair(namespace={v.get_namespace()}, name={v.get_name()}, partitions={v.get_partitions()})"
                )
                self.get_gc.register_clean_action(name, tag, v, "destroy", {})
                rtn[party] = Table(v)
            # others
            else:
                LOGGER.debug(f"[{log_str}] got object with type: {type(v)}")
                rtn[party] = v

        return [rtn[party] for party in parties]

    def push(self, v, name: str, tag: str, parties: List[PartyMeta]):
        if isinstance(v, Table):
            # noinspection PyProtectedMember
            v = v._rp
        log_str = f"federation.eggroll.remote.{name}.{tag}{parties})"
        if v is None:
            raise ValueError(f"[{log_str}]remote `None` to {parties}")
        for party in parties:
            if (name, tag, party) in self._remote_history:
                raise ValueError(f"[{log_str}]remote to {parties} with duplicate tag")
            self._remote_history.add((name, tag, party))

        if isinstance(v, RollPair):
            LOGGER.debug(
                f"[{log_str}]remote "
                f"RollPair(namespace={v.get_namespace()}, name={v.get_name()}, partitions={v.get_partitions()})"
            )
            self.remote_gc.register_clean_action(name, tag, v, "destroy", {})
            _push_with_exception_handle(self._rsc, v, name, tag, parties)
        else:
            LOGGER.debug(f"[{log_str}]remote object with type: {type(v)}")
            _push_with_exception_handle(self._rsc, v, name, tag, parties)

    def destroy(self):
        self._rp_ctx.cleanup(name="*", namespace=self._session_id)


def _push_with_exception_handle(rsc, v, name: str, tag: str, parties: List[PartyMeta]):
    def _remote_exception_re_raise(f, p: PartyMeta):
        try:
            f.result()
            LOGGER.debug(f"[federation.eggroll.remote.{name}.{tag}]future to remote to party: {p} done")
        except Exception as e:
            pid = os.getpid()
            LOGGER.exception(
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
    LOGGER.debug("future `{future}` done, remove")
    _remote_futures.remove(future)


def add_remote_futures(fs: typing.List[concurrent.futures.Future]):
    for f in fs:
        f.add_done_callback(_clear_callback)
        _remote_futures.add(f)


def wait_all_remote_done(timeout=None):
    concurrent.futures.wait(_remote_futures, timeout=timeout, return_when=concurrent.futures.ALL_COMPLETED)
