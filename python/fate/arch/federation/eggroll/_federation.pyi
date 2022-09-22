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

import typing

from eggroll.roll_pair.roll_pair import RollPairContext
from eggroll.roll_site.roll_site import RollSiteContext
from fate_arch.abc import GarbageCollectionABC
from fate_arch.common import Party


class Federation(object):

    def __init__(self, rp_ctx: RollPairContext, rs_session_id: str, party: Party, proxy_endpoint: str):
        self._rsc: RollSiteContext = ...
        ...

    def get(self: Federation, name: str, tag: str, parties: typing.List[Party],
            gc: GarbageCollectionABC) -> typing.List: ...

    def remote(self, v, name: str, tag: str, parties: typing.List[Party],
               gc: GarbageCollectionABC) -> typing.NoReturn: ...


def _remote(v,
            name: str,
            tag: str,
            parties: typing.List[typing.Tuple[str, str]],
            rsc: RollSiteContext,
            gc: GarbageCollectionABC) -> typing.NoReturn: ...


def _get(name: str,
         tag: str,
         parties: typing.List[typing.Tuple[str, str]],
         rsc: RollSiteContext,
         gc: GarbageCollectionABC) -> typing.List: ...


def _remote_tag_not_duplicate(name: str, tag: str, parties: typing.List[typing.Tuple[str, str]]): ...


def _push_with_exception_handle(rsc: RollSiteContext, v, name: str, tag: str, parties: typing.List[typing.Tuple[str, str]]): ...


def _get_tag_not_duplicate(name: str, tag: str, party: typing.Tuple[str, str]): ...


def _get_value_post_process(v, name: str, tag: str, party: typing.Tuple[str, str], rsc: RollSiteContext,
                            gc: GarbageCollectionABC): ...
