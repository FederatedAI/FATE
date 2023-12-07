import os
from dataclasses import dataclass, field
from typing import List

from .argparser import HfArgumentParser, get_parties


@dataclass
class LauncherStandaloneContextArgs:
    federation_session_id: str = field()
    parties: List[str] = field()
    rank: int = field()
    csession_id: str = field(default=None)
    data_dir: str = field(default=None)


@dataclass
class LauncherDistributedContextArgs:
    federation_session_id: str = field()
    parties: List[str] = field()
    rank: int = field()
    config_properties: str = field()
    csession_id: str = field(default=None)
    host: str = field(default="127.0.0.1")
    port: int = field(default=9377)
    federation_mode: str = field(default="message_queue")


@dataclass
class LauncherContextArguments:
    context_type: str = field(default="standalone")


def init_context():
    args = HfArgumentParser(LauncherContextArguments).parse_known_args()[0]
    if args.context_type == "local":
        return init_standalone_context()
    elif args.context_type == "cluster":
        return init_distributed_context()
    else:
        raise ValueError(f"unknown context type: {args.context_type}")


def init_standalone_context():
    from fate.arch.utils.paths import get_base_dir
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation import FederationBuilder
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
    federation_session = FederationBuilder(
        federation_id=args.federation_session_id, party=party, parties=parties
    ).build_standalone(
        computing_session,
    )
    context = Context(computing=computing_session, federation=federation_session)
    return context


def init_distributed_context():
    from fate.arch.federation import FederationBuilder, FederationMode
    from fate.arch.computing.eggroll import CSession

    from fate.arch.context import Context

    args = HfArgumentParser(LauncherDistributedContextArgs).parse_args_into_dataclasses(return_remaining_strings=True)[
        0
    ]
    parties = get_parties(args.parties)
    party = parties[args.rank]
    computing_session = CSession(session_id=args.csession_id, config_properties_file=args.config_properties)
    federation_mode = FederationMode.from_str(args.federation_mode)

    federation_session = FederationBuilder(
        federation_id=args.federation_session_id, party=party, parties=parties
    ).build_osx(
        computing_session=computing_session,
        host=args.host,
        port=args.port,
        mode=federation_mode,
    )
    context = Context(computing=computing_session, federation=federation_session)
    return context
