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

import os
from dataclasses import dataclass, field
from typing import List

from .argparser import HfArgumentParser, get_parties


@dataclass
class LauncherLocalContextArgs:
    parties: List[str] = field()
    rank: int = field()
    csession_id: str = field(default=None)
    data_dir: str = field(default=None)


@dataclass
class LauncherClusterContextArgs:
    parties: List[str] = field()
    rank: int = field()
    config_properties: str = field()
    csession_id: str = field(default=None)
    federation_address: str = field(default="127.0.0.1:9377")
    cluster_address: str = field(default="127.0.0.1:4670")
    federation_mode: str = field(default="stream")


@dataclass
class LauncherContextArguments:
    context_type: str = field(default="local")


def init_context(computing_session_id: str, federation_session_id: str):
    args = HfArgumentParser(LauncherContextArguments).parse_known_args()[0]
    if args.context_type == "local":
        return init_local_context(computing_session_id, federation_session_id)
    elif args.context_type == "cluster":
        return init_cluster_context(computing_session_id, federation_session_id)
    else:
        raise ValueError(f"unknown context type: {args.context_type}")


def init_local_context(computing_session_id: str, federation_session_id: str):
    from .paths import get_base_dir
    from fate.arch.computing.backends.standalone import CSession
    from fate.arch.federation import FederationBuilder
    from fate.arch.context import Context

    args = HfArgumentParser(LauncherLocalContextArgs).parse_args_into_dataclasses(return_remaining_strings=True)[0]

    data_dir = args.data_dir
    if not data_dir:
        data_dir = os.path.join(get_base_dir(), "data")
    computing_session = CSession(session_id=computing_session_id, data_dir=data_dir)

    parties = get_parties(args.parties)
    party = parties[args.rank]
    federation_session = FederationBuilder(
        federation_session_id=federation_session_id, party=party, parties=parties
    ).build_standalone(
        computing_session,
    )
    context = Context(computing=computing_session, federation=federation_session)
    return context


def init_cluster_context(computing_session_id: str, federation_session_id: str):
    from fate.arch.federation import FederationBuilder, FederationMode
    from fate.arch.computing.backends.eggroll import CSession

    from fate.arch.context import Context

    args = HfArgumentParser(LauncherClusterContextArgs).parse_args_into_dataclasses(return_remaining_strings=True)[0]

    cluster_host, cluster_port = args.cluster_address.split(":")
    computing_session = CSession(
        session_id=computing_session_id,
        host=cluster_host.strip(),
        port=int(cluster_port.strip()),
    )

    parties = get_parties(args.parties)
    party = parties[args.rank]
    federation_mode = FederationMode.from_str(args.federation_mode)
    federation_host, federation_port = args.federation_address.split(":")
    federation_session = FederationBuilder(
        federation_session_id=federation_session_id, party=party, parties=parties
    ).build_osx(
        computing_session=computing_session,
        host=federation_host.strip(),
        port=int(federation_port.strip()),
        mode=federation_mode,
    )

    context = Context(computing=computing_session, federation=federation_session)
    return context
