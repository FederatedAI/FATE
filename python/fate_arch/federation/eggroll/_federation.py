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
import functools
import os
import signal
from enum import Enum

from eggroll.roll_pair.roll_pair import RollPair
from eggroll.roll_site.roll_site import RollSiteContext
from fate_arch.abc import FederationABC
from fate_arch.common.log import getLogger
from fate_arch.computing.eggroll import Table
from fate_arch.federation._split import _is_split_head, _split_get, _is_splitable_obj, _get_splits

LOGGER = getLogger()


class Federation(FederationABC):

    def __init__(self, rp_ctx, rs_session_id, party, proxy_endpoint):
        LOGGER.debug(f"init eggroll federation: rp_session_id={rp_ctx.session_id}, rs_session_id={rs_session_id},"
                     f"party={party}, proxy_endpoint={proxy_endpoint}")

        from eggroll.core.meta_model import ErEndpoint
        if isinstance(proxy_endpoint, str):
            splited = proxy_endpoint.split(':')
            proxy_endpoint = ErEndpoint(host=splited[0].strip(), port=int(splited[1].strip()))

        options = {
            'self_role': party.role,
            'self_party_id': party.party_id,
            'proxy_endpoint': proxy_endpoint
        }
        self._rsc = RollSiteContext(rs_session_id, rp_ctx=rp_ctx, options=options)
        LOGGER.debug(f"init eggroll federation context done")

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


def _remote(v, name, tag, parties, rsc, gc):
    log_str = f"federation.remote(name={name}, tag={tag}, parties={parties})"
    assert v is not None, \
        f"[{log_str}]remote `None`"
    assert _remote_tag_not_duplicate(name, tag, parties), \
        f"[{log_str}]duplicate tag"
    LOGGER.debug(f"[{log_str}]remote data, type={type(v)}")

    t = _get_type(v)
    if t == _FederationValueType.ROLL_PAIR:
        gc.add_gc_action(tag, v, 'destroy', {})
        _push_with_exception_handle(rsc, v, name, tag, parties)
        return

    if t == _FederationValueType.SPLIT_OBJECT:
        head, tails = _get_splits(v)
        _push_with_exception_handle(rsc, head, name, tag, parties)

        for k, tail in enumerate(tails):
            _push_with_exception_handle(rsc, tail, name, f"{tag}.__part_{k}", parties)

        return

    if t == _FederationValueType.OBJECT:
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
        rtn[party] = _get_value_post_process(v, name, tag, party, rsc, gc)
    return [rtn[party] for party in parties]


class _FederationValueType(Enum):
    OBJECT = 1
    ROLL_PAIR = 2
    SPLIT_OBJECT = 3


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
    if _is_splitable_obj(v):
        return _FederationValueType.SPLIT_OBJECT
    return _FederationValueType.OBJECT


def _push_with_exception_handle(rsc, v, name, tag, parties):
    def _remote_exception_re_raise(f, p):
        try:
            f.result()
            LOGGER.debug(f"[name={name}, tag={tag}, party={p}]remote done")
        except Exception as e:
            pid = os.getpid()
            LOGGER.exception(f"remote fail, terminating process(pid={pid})")
            os.kill(pid, signal.SIGTERM)
            raise e

    def _callback(p):
        return functools.partial(_remote_exception_re_raise, p=p)

    rs = rsc.load(name=name, tag=tag)
    futures = rs.push(obj=v, parties=parties)
    for party, future in zip(parties, futures):
        future.add_done_callback(_callback(party))
    return rs


_get_history = set()


def _get_tag_not_duplicate(name, tag, party):
    if (name, tag, party) in _get_history:
        return False
    _get_history.add((name, tag, party))
    return True


def _get_value_post_process(v, name, tag, party, rsc, gc):
    log_str = f"federation.get(name={name}, tag={tag}, party={party})"
    assert v is not None, \
        f"[{log_str}]get None"
    assert _get_tag_not_duplicate(name, tag, party), \
        f"[{log_str}]duplicate tag"
    LOGGER.debug(f"[{log_str}]got data, type={type(v)}]")

    # got a roll pair
    if isinstance(v, RollPair):
        gc.add_gc_action(tag, v, 'destroy', {})
        return v

    # got large object in splits
    if _is_split_head(v):
        num_split = v.num_split()
        LOGGER.debug(f"[{log_str}]is split object, num_split={num_split}")
        split_objs = []
        for k in range(num_split):
            split_obj = _split_rs = rsc.load(name, tag=f"{tag}.__part_{k}").pull([party])[0].result()
            LOGGER.debug(f"[{log_str}]got split ({k}/{num_split})")
            split_objs.append(split_obj)
        obj = _split_get(split_objs)
        return obj

    # others
    return v
