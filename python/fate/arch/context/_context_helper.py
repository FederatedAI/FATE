from typing import Tuple, List

from fate.arch.computing import ComputingBuilder, ComputingEngine
from fate.arch.context import Context
from fate.arch.federation import FederationBuilder, FederationType


def create_context(
    local_party: Tuple[str, str],
    parties: List[Tuple[str, str]],
    federation_session_id,
    federation_engine=FederationType.STANDALONE,
    federation_conf: dict = None,
    computing_session_id=None,
    computing_engine=ComputingEngine.STANDALONE,
    computing_conf=None,
):
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
        federation_id=federation_session_id,
        party=local_party,
        parties=parties,
    ).build(computing_session, federation_engine, federation_conf)
    return Context(computing=computing_session, federation=federation_session)
