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
import os
import signal
from enum import Enum

from eggroll.roll_pair.roll_pair import RollPair
from eggroll.roll_site.roll_site import RollSiteContext
from fate_arch.abc import FederationABC
from fate_arch.common.log import getLogger
from fate_arch.computing.eggroll import Table
from fate_arch.common import remote_status

LOGGER = getLogger()


class Federation(FederationABC):
    def __init__(self, rp_ctx, rs_session_id, party, proxy_endpoint):
        LOGGER.debug(
            f"[federation.eggroll]init federation: "
            f"rp_session_id={rp_ctx.session_id}, rs_session_id={rs_session_id}, "
            f"party={party}, proxy_endpoint={proxy_endpoint}"
        )

        options = {
            "self_role": party.role,
            "self_party_id": party.party_id,
            "proxy_endpoint": proxy_endpoint,
        }
        self._session_id = rs_session_id
        self._rp_ctx = rp_ctx
        self._rsc = RollSiteContext(rs_session_id, rp_ctx=rp_ctx, options=options)
        LOGGER.debug(f"[federation.eggroll]init federation context done")

    @property
    def session_id(self) -> str:
        return self._session_id

    def get(self, name, tag, parties, gc):
        parties = [(party.role, party.party_id) for party in parties]
        raw_result = _get(name, tag, parties, self._rsc, gc)
        return [Table(v) if isinstance(v, RollPair) else v for v in raw_result]

    def remote(self, v, name, tag, parties, gc):
        if isinstance(v, Table):
            # noinspection PyProtectedMember
            v = v._rp
        parties = [(party.role, party.party_id) for party in parties]
        _remote(v, name, tag, parties, self._rsc, gc)

    def destroy(self, parties):
        self._rp_ctx.cleanup(name="*", namespace=self._session_id)


def _remote(v, name, tag, parties, rsc, gc):
    log_str = f"federation.eggroll.remote.{name}.{tag}{parties})"
    if v is None:
        raise ValueError(f"[{log_str}]remote `None` to {parties}")
    if not _remote_tag_not_duplicate(name, tag, parties):
        raise ValueError(f"[{log_str}]remote to {parties} with duplicate tag")

    t = _get_type(v)
    if t == _FederationValueType.ROLL_PAIR:
        LOGGER.debug(
            f"[{log_str}]remote "
            f"RollPair(namespace={v.get_namespace()}, name={v.get_name()}, partitions={v.get_partitions()})"
        )
        gc.add_gc_action(tag, v, "destroy", {})
        _push_with_exception_handle(rsc, v, name, tag, parties)
        return

    if t == _FederationValueType.OBJECT:
        LOGGER.debug(f"[{log_str}]remote object with type: {type(v)}")
        _push_with_exception_handle(rsc, v, name, tag, parties)
        return

    raise NotImplementedError(f"t={t}")


def _get(name, tag, parties, rsc, gc):
    rs = rsc.load(name=name, tag=tag)
    future_map = dict(zip(rs.pull(parties=parties), parties))
    rtn = {}
    for future in concurrent.futures.as_completed(future_map):
        party = future_map[future]
        v = future.result()
        rtn[party] = _get_value_post_process(v, name, tag, party, gc)
    return [rtn[party] for party in parties]


class _FederationValueType(Enum):
    OBJECT = 1
    ROLL_PAIR = 2


_remote_history = set()


def _remote_tag_not_duplicate(name, tag, parties):
    for party in parties:
        if (name, tag, party) in _remote_history:
            return False
        _remote_history.add((name, tag, party))
    return True


def _get_type(v):
    if isinstance(v, RollPair):
        return _FederationValueType.ROLL_PAIR
    return _FederationValueType.OBJECT


def _push_with_exception_handle(rsc, v, name, tag, parties):
    def _remote_exception_re_raise(f, p):
        try:
            f.result()
            LOGGER.debug(
                f"[federation.eggroll.remote.{name}.{tag}]future to remote to party: {p} done"
            )
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

    def _get_call_back_func(p):
        def _callback(f):
            return _remote_exception_re_raise(f, p)

        return _callback

    rs = rsc.load(name=name, tag=tag)
    futures = rs.push(obj=v, parties=parties)
    for party, future in zip(parties, futures):
        future.add_done_callback(_get_call_back_func(party))

    remote_status.add_remote_futures(futures)
    return rs


_get_history = set()


def _get_tag_not_duplicate(name, tag, party):
    if (name, tag, party) in _get_history:
        return False
    _get_history.add((name, tag, party))
    return True


def _get_value_post_process(v, name, tag, party, gc):
    log_str = f"federation.eggroll.get.{name}.{tag}"
    if v is None:
        raise ValueError(f"[{log_str}]get `None` from {party}")
    if not _get_tag_not_duplicate(name, tag, party):
        raise ValueError(f"[{log_str}]get from {party} with duplicate tag")

    # got a roll pair
    if isinstance(v, RollPair):
        LOGGER.debug(
            f"[{log_str}] got "
            f"RollPair(namespace={v.get_namespace()}, name={v.get_name()}, partitions={v.get_partitions()})"
        )
        gc.add_gc_action(tag, v, "destroy", {})
        return v
    # others
    LOGGER.debug(f"[{log_str}] got object with type: {type(v)}")
    return v
