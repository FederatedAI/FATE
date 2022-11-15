import enum

from .artifacts import *
from .spec import *
from .types import *


class Artifact(enum.Enum):
    TrainData = TrainData
    ValidateData = ValidateData
    TestData = TestData
    TrainOutputData = TrainOutputData
    TestOutputData = TestOutputData
    Model = Model
    Metrics = Metrics
