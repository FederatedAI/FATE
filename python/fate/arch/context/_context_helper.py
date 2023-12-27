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

from typing import Tuple, List, Optional

from fate.arch.computing import ComputingBuilder, ComputingEngine
from fate.arch.context import Context
from fate.arch.federation import FederationBuilder, FederationEngine


def create_context(
    local_party: Tuple[str, str],
    parties: List[Tuple[str, str]],
    federation_session_id: str,
    federation_engine: FederationEngine = FederationEngine.STANDALONE,
    federation_conf: dict = None,
    computing_session_id: Optional[str] = None,
    computing_engine: ComputingEngine = ComputingEngine.STANDALONE,
    computing_conf: dict = None,
):
    f"""
    helper function to create context, especially for direct use in ml

    Args:
        local_party: local party, (role, party_id) tuple
        parties: all parties, [(role, party_id), ...]
        federation_session_id: federation session id
        federation_engine: federation engine, default to standalone
        federation_conf: federation conf, different engine has different conf
        computing_session_id: computing session id, if not set, will be `<federation_session_id>_<local_party[0]>_<local_party[1]>`
        computing_engine: computing engine, default to standalone
        computing_conf: computing conf, different engine has different conf

    Returns:
        Context

    Examples:
        below is an example to create context for standalone computing and standalone federation
        
        >>> import uuid
        >>> from fate.arch.context import create_context
        >>> context = create_context(
        >>>    local_party=("guest", "9999"), 
        >>>    parties=[("guest", "9999"), ("host", "10000")], 
        >>>    federation_session_id=str(uuid.uuid1()))
    """
    if federation_conf is None:
        federation_conf = {}
    if computing_conf is None:
        computing_conf = {}
    if ComputingEngine.STANDALONE == computing_engine:
        if "computing.standalone.data_dir" not in computing_conf:
            computing_conf["computing.standalone.data_dir"] = "/tmp"
    if computing_session_id is None:
        computing_session_id = f"{federation_session_id}_{local_party[0]}_{local_party[1]}"
    computing_session = ComputingBuilder(computing_session_id=computing_session_id).build(
        computing_engine, computing_conf
    )
    federation_session = FederationBuilder(
        federation_session_id=federation_session_id,
        party=local_party,
        parties=parties,
    ).build(computing_session, federation_engine, federation_conf)
    return Context(computing=computing_session, federation=federation_session)
