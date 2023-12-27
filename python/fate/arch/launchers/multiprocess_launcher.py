# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

# Note: This file is modified from crypton

import datetime
import logging
import multiprocessing
import signal
import sys
import time
import uuid
from argparse import Namespace
from dataclasses import dataclass, field
from multiprocessing import Queue, Event
from typing import List

import rich
import rich.console
import rich.panel
import rich.traceback

from fate.arch import trace
from .argparser import HfArgumentParser

logger = logging.getLogger(__name__)


class MultiProcessLauncher:
    def __init__(
        self,
        console: rich.console.Console,
        parties: List[str] = None,
        federation_session_id: str = None,
        data_dir: str = None,
        log_level: str = None,
    ):
        namespace = Namespace()
        if federation_session_id is not None:
            namespace.federation_session_id = federation_session_id
        if parties is not None:
            namespace.parties = parties
        if data_dir is not None:
            namespace.data_dir = data_dir
        if log_level is not None:
            namespace.log_level = log_level
        args = HfArgumentParser(LauncherArguments).parse_known_args(namespace=namespace)[0]

        multiprocessing.set_start_method("spawn")
        self.world_size = len(args.parties)
        self.processes: List[multiprocessing.Process] = []
        self.output_or_exception_q = Queue()
        self.safe_to_exit = Event()  # barrier
        self.console = console
        self._exception_tb = {}
        self.federation_session_id = args.federation_session_id

    def start(self, f, carrier=None):
        if carrier is None:
            carrier = {}
        for rank in range(self.world_size):
            process_name = "process " + str(rank)
            output_or_exception_q = self.output_or_exception_q
            safe_to_exit = self.safe_to_exit
            width = self.console.width
            argv = sys.argv.copy()
            argv.extend(["--rank", str(rank)])
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(
                    carrier,
                    output_or_exception_q,
                    safe_to_exit,
                    width,
                    self.federation_session_id,
                    argv,
                    f,
                ),
            )
            self.processes.append(process)

        from fate.arch.protocol import mpc

        # TODO: not work for now, need to fix
        if mpc.ttp_required():
            ttp_process = multiprocessing.Process(
                target=self.__class__._run_ttp_server,
                name="TTP",
                args=(self.world_size,),
            )
            self.processes.append(ttp_process)

        for process in self.processes:
            process.start()

    @classmethod
    def _run_ttp_server(cls, world_size):
        from fate.arch.protocol import mpc

        mpc.provider.TTPServer()

    @classmethod
    def _run_process(
        cls,
        carrier: dict,
        output_or_exception_q: Queue,
        safe_to_exit: Event,
        width,
        federation_session_id,
        argv,
        f,
    ):
        sys.argv = argv
        args = HfArgumentParser(LauncherProcessArguments).parse_args_into_dataclasses(return_remaining_strings=True)[0]
        from fate.arch.launchers.logger import set_up_logging
        from fate.arch.launchers.context_helper import init_context
        from fate.arch.trace import setup_tracing
        from fate.arch.trace import profile_start, profile_ends

        if args.rank >= len(args.parties):
            raise ValueError(f"rank {args.rank} is out of range {len(args.parties)}")
        parties = args.get_parties()
        party = parties[args.rank]
        csession_id = f"{federation_session_id}_{party[0]}_{party[1]}"

        # set up logging
        set_up_logging(args.rank, args.log_level)
        logger = logging.getLogger(__name__)

        # set up tracing
        setup_tracing(f"fate:{party[0]}-{party[1]}")
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span(name=csession_id, context=trace.extract_carrier(carrier)) as span:
            ctx = init_context(computing_session_id=csession_id, federation_session_id=federation_session_id)

            try:
                profile_start()
                f(ctx)
                profile_ends()
                output_or_exception_q.put((args.rank, None, None))
                safe_to_exit.wait()

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                logger.error(f"exception in rank {args.rank}: {e}", stack_info=True)
                exc_traceback = rich.traceback.Traceback.from_exception(
                    type(e), e, traceback=e.__traceback__, width=width, show_locals=True
                )
                try:
                    output_or_exception_q.put((args.rank, e, exc_traceback))
                except Exception as e:
                    logger.exception(f"failed to put exception to queue: {e}")
            finally:
                try:
                    ctx.destroy()
                except Exception:
                    pass

    def wait(self) -> int:
        logger = logging.getLogger(__name__)

        uncompleted_ranks = set(range(len(self.processes)))
        for i in range(len(self.processes)):
            rank, e, exc_traceback = self.output_or_exception_q.get()
            uncompleted_ranks.remove(rank)
            if e is not None:
                self._exception_tb[rank] = exc_traceback
                return 1
            else:
                logger.info(f"rank {rank} exited successfully, waiting for other ranks({uncompleted_ranks}) to exit")
        else:
            return 0

    def terminate(self):
        self.safe_to_exit.set()
        time.sleep(10)  # wait for 1 second to let all processes has a chance to exit
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join()
        self.output_or_exception_q.close()

    def show_exceptions(self):
        for rank, tb in self._exception_tb.items():
            self.console.print(rich.panel.Panel(tb, title=f"rank {rank} exception", expand=False, border_style="red"))

    def block_run(self, f):
        from fate.arch.trace import setup_tracing

        setup_tracing("multi_process_launcher")
        with trace.get_tracer(__name__).start_as_current_span(self.federation_session_id):
            carrier = trace.inject_carrier()
            self.start(f, carrier=carrier)

            def sigterm_handler(signum, frame):
                self.terminate()
                exit(1)

            signal.signal(signal.SIGTERM, sigterm_handler)
            logger.info("waiting for all processes to exit")
            exit_code = self.wait()
            logger.info("all processes exited")
            logger.info("cleaning up")
            self.terminate()
            logger.info("done")
            if exit_code != 0:
                self.show_exceptions()
            exit(exit_code)


@dataclass
class LauncherArguments:
    parties: List[str] = field(metadata={"required": True})
    federation_session_id: str = field(
        default_factory=lambda: f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid1().hex[:6]}"
    )
    tracer_id: str = field(default_factory=lambda: uuid.uuid1().hex[:6])
    data_dir: str = field(default=None)
    log_level: str = field(default="INFO")


@dataclass
class LauncherProcessArguments:
    log_level: str = field()
    rank: int = field()
    parties: List[str] = field(metadata={"required": True})

    def get_parties(self):
        parties = []
        for party in self.parties:
            if isinstance(party, str):
                parties.append(tuple(party.split(":")))
            else:
                parties.append(party)
        return parties


def launch(f, **kwargs):
    namespace = Namespace()
    if "federation_session_id" in kwargs:
        namespace.federation_session_id = kwargs["federation_session_id"]
    if "parties" in kwargs:
        namespace.parties = kwargs["parties"]
    if "data_dir" in kwargs:
        namespace.data_dir = kwargs["data_dir"]
    if "log_level" in kwargs:
        namespace.log_level = kwargs["log_level"]

    args_desc = [LauncherArguments]
    args_desc.extend(kwargs.get("extra_args_desc", []))
    args, _ = HfArgumentParser(args_desc).parse_known_args(namespace=namespace)

    from fate.arch.launchers.logger import set_up_logging

    set_up_logging(-1, args.log_level)

    logger.info("========================================================")
    logger.info(f"federation id: {args.federation_session_id}")
    logger.info(f"parties: {args.parties}")
    logger.info(f"data dir: {args.data_dir}")
    logger.info("========================================================")
    console = rich.console.Console()
    launcher = MultiProcessLauncher(console, args.parties, args.federation_session_id, args.data_dir, args.log_level)
    launcher.block_run(f)
