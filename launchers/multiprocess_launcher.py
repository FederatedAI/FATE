#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import datetime
import uuid
import multiprocessing
import click


class MultiProcessLauncher:
    def __init__(self, world_size, parties, federation_session_id, proc, data_dir, log_level, parameters):
        multiprocessing.set_start_method("spawn")
        self.processes = []
        for rank in range(world_size):
            process_name = "process " + str(rank)
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(rank, parties, federation_session_id, proc, data_dir, log_level, parameters),
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
    def _run_process(cls, rank, parties, federation_session_id, proc, data_dir, log_level, parameters):
        from fate.arch.utils.logger import set_up_logging
        from fate.arch.utils.context_helper import init_standalone_context

        # set up logging
        set_up_logging(rank, log_level)

        # init context
        parties = [tuple(p.split(":")) for p in parties]
        if rank >= len(parties):
            raise ValueError(f"rank {rank} is out of range {len(parties)}")
        party = parties[rank]
        csession_id = f"{federation_session_id}_{party[0]}_{party[1]}"
        ctx = init_standalone_context(csession_id, federation_session_id, party, parties, data_dir)

        # init crypten
        from fate.ml.mpc import MPCModule

        ctx.mpc.init()

        # get proc cls
        module_name, cls_name = proc.split(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        assert issubclass(cls, MPCModule), f"{cls} is not a subclass of MPCModule"
        parameters = cls.parse_parameters(parameters)
        inst = cls(**parameters)
        inst.fit(ctx)

    def start(self):
        for process in self.processes:
            process.start()

    def join(self):
        for process in self.processes:
            process.join()
            assert process.exitcode == 0, f"{process.name} has non-zero exit code {process.exitcode}"

    def terminate(self):
        for process in self.processes:
            process.terminate()


@click.command()
@click.option("--federation_session_id", type=str, help="federation session id")
@click.option("--parties", multiple=True, type=str, help="parties", required=True)
@click.option("--data_dir", type=str, help="data dir")
@click.option("--proc", type=str, help="proc, e.g. fate.ml.mpc.svm:SVM", required=True)
@click.option("--log_level", type=str, help="log level", default="INFO")
@click.option("-p", "--parameter", multiple=True, type=str, help="parameters")
def cli(federation_session_id, parties, data_dir, proc, log_level, parameter):
    if not federation_session_id:
        federation_session_id = f"{datetime.datetime.now().strftime('YYMMDD-hh:mm-ss')}-{uuid.uuid1()}"
    parameters = {}
    for p in parameter:
        k, v = p.split("=")
        parameters[k] = v
    print("========================================================")
    print(f"federation id: {federation_session_id}")
    print(f"parties: {parties}")
    print(f"data dir: {data_dir}")
    print(f"proc: {proc}")
    print("========================================================")
    launcher = MultiProcessLauncher(len(parties), parties, federation_session_id, proc, data_dir, log_level, parameters)
    launcher.start()
    launcher.join()
    launcher.terminate()


if __name__ == "__main__":
    cli()
