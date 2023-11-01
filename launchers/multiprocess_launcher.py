#!/usr/bin/env python3
import time

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import signal
import rich
import rich.console
import rich.panel
import rich.traceback
import logging
import importlib
import datetime
import uuid
import multiprocessing
from multiprocessing import Queue, Event
import click

logger = logging.getLogger()


class MultiProcessLauncher:
    def __init__(
        self,
        console: rich.console.Console,
        world_size,
        parties,
        federation_session_id,
        proc,
        data_dir,
        log_level,
        parameters,
    ):
        multiprocessing.set_start_method("spawn")
        self.processes: List[multiprocessing.Process] = []
        self.output_or_exception_q = Queue()
        self.safe_to_exit = Event()  # barrier
        self._console = console
        self._exception_tb = {}
        for rank in range(world_size):
            process_name = "process " + str(rank)
            output_or_exception_q = self.output_or_exception_q
            safe_to_exit = self.safe_to_exit
            width = console.width
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(
                    output_or_exception_q,
                    safe_to_exit,
                    width,
                    rank,
                    parties,
                    federation_session_id,
                    proc,
                    data_dir,
                    log_level,
                    parameters,
                ),
            )
            self.processes.append(process)

        # if crypten.mpc.ttp_required():
        #     ttp_process = multiprocessing.Process(
        #         target=self.__class__._run_process,
        #         name="TTP",
        #         args=(
        #             world_size,
        #             world_size,
        #             env,
        #             crypten.mpc.provider.TTPServer,
        #             None,
        #         ),
        #     )
        #     self.processes.append(ttp_process)

    @classmethod
    def _run_process(
        cls,
        output_or_exception_q: Queue,
        safe_to_exit: Event,
        width,
        rank,
        parties,
        federation_session_id,
        proc,
        data_dir,
        log_level,
        parameters,
    ):
        from fate.arch.utils.logger import set_up_logging
        from fate.arch.utils.context_helper import init_standalone_context

        # set up logging
        set_up_logging(rank, log_level)
        logger = logging.getLogger()

        # init context
        parties = [tuple(p.split(":")) for p in parties]
        if rank >= len(parties):
            raise ValueError(f"rank {rank} is out of range {len(parties)}")
        party = parties[rank]
        csession_id = f"{federation_session_id}_{party[0]}_{party[1]}"
        ctx = init_standalone_context(csession_id, federation_session_id, party, parties, data_dir)

        try:
            # init crypten
            from fate.ml.mpc import MPCModule

            ctx.mpc.init()

            # get proc cls
            module_name, cls_name = proc.split(":")
            module = importlib.import_module(module_name)
            mpc_module = getattr(module, cls_name)
            assert issubclass(mpc_module, MPCModule), f"{mpc_module} is not a subclass of MPCModule"
            parameters = mpc_module.parse_parameters(parameters)
            inst = mpc_module(**parameters)
            inst.fit(ctx)
            output_or_exception_q.put((rank, None, None))
            safe_to_exit.wait()

        except Exception as e:
            logger.error(f"exception in rank {rank}: {e}", stack_info=False)
            exc_traceback = rich.traceback.Traceback.from_exception(
                type(e), e, traceback=e.__traceback__, width=width, show_locals=True
            )
            output_or_exception_q.put((rank, e, exc_traceback))
        finally:
            try:
                ctx.destroy()
            except Exception:
                pass

    def start(self):
        for process in self.processes:
            process.start()

    def wait(self) -> int:
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
        time.sleep(1)  # wait for 1 second to let all processes has a chance to exit
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join()
        self.output_or_exception_q.close()

    def show_exceptions(self):
        for rank, tb in self._exception_tb.items():
            self._console.print(rich.panel.Panel(tb, title=f"rank {rank} exception", expand=False, border_style="red"))


@click.command()
@click.option("--federation_session_id", type=str, help="federation session id")
@click.option("--parties", multiple=True, type=str, help="parties", required=True)
@click.option("--data_dir", type=str, help="data dir")
@click.option("--proc", type=str, help="proc, e.g. fate.ml.mpc.svm:SVM", required=True)
@click.option("--log_level", type=str, help="log level", default="INFO")
@click.option("-p", "--parameter", multiple=True, type=str, help="parameters")
def cli(federation_session_id, parties, data_dir, proc, log_level, parameter):
    from fate.arch.utils.logger import set_up_logging

    set_up_logging(-1, log_level)
    if not federation_session_id:
        federation_session_id = f"{datetime.datetime.now().strftime('YYMMDD-hh:mm-ss')}-{uuid.uuid1()}"
    parameters = {}
    for p in parameter:
        k, v = p.split("=")
        parameters[k] = v
    logger.info("========================================================")
    logger.info(f"federation id: {federation_session_id}")
    logger.info(f"parties: {parties}")
    logger.info(f"data dir: {data_dir}")
    logger.info(f"proc: {proc}")
    logger.info("========================================================")
    console = rich.console.Console()
    launcher = MultiProcessLauncher(
        console, len(parties), parties, federation_session_id, proc, data_dir, log_level, parameters
    )
    launcher.start()

    def sigterm_handler(signum, frame):
        launcher.terminate()
        exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)
    logger.info("waiting for all processes to exit")
    exit_code = launcher.wait()
    logger.info("all processes exited")
    logger.info("cleaning up")
    launcher.terminate()
    logger.info("done")
    if exit_code != 0:
        launcher.show_exceptions()
    exit(exit_code)


if __name__ == "__main__":
    cli()
