import os
from typing import List
from dataclasses import dataclass, field
from .argparser import HfArgumentParser, get_parties


@dataclass
class LauncherStandaloneContextArgs:
    federation_session_id: str = field()
    parties: List[str] = field()
    rank: int = field()
    csession_id: str = field(default=None)
    data_dir: str = field(default=None)


@dataclass
class LauncherEggrollContextArgs:
    federation_session_id: str = field()
    parties: List[str] = field()
    rank: int = field()
    csession_id: str = field(default=None)
    host: str = field(default="127.0.0.1")
    port: int = field(default=9377)


@dataclass
class LauncherContextArguments:
    context_type: str = field(default="standalone")


def init_context():
    args = HfArgumentParser(LauncherContextArguments).parse_known_args()[0]
    if args.context_type == "standalone":
        return init_standalone_context()
    elif args.context_type == "eggroll":
        return init_eggroll_context()
    else:
        raise ValueError(f"unknown context type: {args.context_type}")


def init_standalone_context():
    from fate.arch.utils.paths import get_base_dir
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    from fate.arch.context import Context

    args = HfArgumentParser(LauncherStandaloneContextArgs).parse_args_into_dataclasses(return_remaining_strings=True)[
        0
    ]

    data_dir = args.data_dir
    if not data_dir:
        data_dir = os.path.join(get_base_dir(), "data")
    computing_session = CSession(session_id=args.csession_id, data_dir=data_dir)
    parties = get_parties(args.parties)
    party = parties[args.rank]
    federation_session = StandaloneFederation(computing_session, args.federation_session_id, party, parties)
    context = Context(computing=computing_session, federation=federation_session)
    return context


def init_eggroll_context():
    from fate.arch.computing.eggroll import CSession

    from fate.arch.federation.osx import OSXFederation
    from fate.arch.federation.eggroll import EggrollFederation
    from fate.arch.context import Context

    args = HfArgumentParser(LauncherEggrollContextArgs).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    parties = get_parties(args.parties)
    party = parties[args.rank]
    computing_session = CSession(session_id=args.csession_id)
    federation_session = EggrollFederation(
        rp_ctx=computing_session.get_rpc(),
        rs_session_id=args.federation_session_id,
        party=party,
        parties=parties,
        proxy_endpoint=f"{args.host}:{args.port}",
    )
    # federation_session = OSXFederation.from_conf(
    #     federation_session_id=args.federation_session_id,
    #     computing_session=computing_session,
    #     party=party,
    #     parties=parties,
    #     host=args.host,
    #     port=args.port,
    # )
    context = Context(computing=computing_session, federation=federation_session)
    return context
