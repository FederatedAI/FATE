from typing import Optional

from fate.components.runner.parser.model import PBModelsLoader
from fate.interface import CheckpointManager as CheckpointManagerInterface


class CheckpointManager(CheckpointManagerInterface):
    def __init__(self, checkpoint_manager) -> None:
        self.checkpoint_manager = checkpoint_manager

    def get_latest_checkpoint(self) -> Optional[PBModelsLoader]:
        if (checkpoint := self.checkpoint_manager.latest_checkpoing) is not None:
            return PBModelsLoader(checkpoint)

    @classmethod
    def parse(cls, checkpoint_manager):
        return CheckpointManager(checkpoint_manager)
