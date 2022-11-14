from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel
from typing_extensions import Annotated, Literal


class Artifact:
    type: str
    ...


class Artifacts(List):
    ...


class DatasetArtifact(Artifact):
    type = "fate.dataset"


class DatasetsArtifact(Artifacts):
    type = "fate.datasets"


class ModelArtifact(Artifact):
    type = "fate.model"


class MetricArtifact(Artifact):
    type = "fate.metric"


class ParameterSpec(BaseModel):
    type: str
    default: Any
    optional: bool


class ArtifactSpec(BaseModel):
    type: str
    optional: bool
    stages: Optional[List[str]]
    roles: Optional[List[Literal["guest", "host", "arbiter"]]]


class InputDefinitionsSpec(BaseModel):
    parameters: Dict[str, ParameterSpec]
    artifacts: Dict[str, ArtifactSpec]


class OutputDefinitionsSpec(BaseModel):
    artifacts: Dict[str, ArtifactSpec]


class ComponentSpec(BaseModel):
    name: str
    description: str
    provider: str
    version: str
    labels: List[str] = ["trainable"]
    roles: List[Literal["guest", "host", "arbiter"]]
    inputDefinitions: InputDefinitionsSpec
    outputDefinitions: OutputDefinitionsSpec


class ComponentSpecV1(BaseModel):
    component: ComponentSpec
    schemaVersion: str = "v1"


class OutputAnnotated:
    ...


class InputAnnotated:
    ...


T = TypeVar("T")
Output = Annotated[T, OutputAnnotated]
Input = Annotated[T, InputAnnotated]

TrainData = Annotated[Input[DatasetArtifact], "trainData"]
ValidateData = Annotated[Input[DatasetArtifact], "validateData"]
TestData = Annotated[Input[DatasetArtifact], "testData"]
TrainOutputData = Annotated[Output[DatasetArtifact], "trainOutputData"]
TestOutputData = Annotated[Output[DatasetArtifact], "testOutputData"]
Model = Annotated[Output[ModelArtifact], "model"]
Metrics = Annotated[Output[MetricArtifact], "metrics"]
ArtifactType = Union[
    TrainData, ValidateData, TestData, TrainOutputData, TestOutputData, Model, Metrics
]


class Cpn:
    def __init__(
        self,
        name: str,
        roles: List[Literal["guest", "host", "arbiter"]],
        provider: str,
        version: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.provider = provider
        self.version = version
        self.description = description
        self.roles = roles

        self._params: Dict[str, ParameterSpec] = {}
        self._artifacts: Dict[str, Tuple[bool, ArtifactSpec]] = {}

    def parameter(
        self,
        name: str,
        type: type,
        default: Any,
        optional: bool,
    ) -> Callable[[T], T]:
        self._params[name] = ParameterSpec(
            type=type.__name__, default=default, optional=optional
        )

        def _wrap(func):
            return func

        return _wrap

    def artifact(
        self,
        name: str,
        type: Type[ArtifactType],
        optional=False,
        roles=None,
        stages=None,
    ) -> Callable[[T], T]:
        annotated, type_name, *_ = getattr(type, "__metadata__", [None, {}])
        name = type_name if type_name else name
        if annotated == OutputAnnotated:
            self._artifacts[name] = (
                True,
                ArtifactSpec(
                    type=type.type, optional=optional, roles=roles, stages=stages
                ),
            )
        elif annotated == InputAnnotated:
            self._artifacts[name] = (
                False,
                ArtifactSpec(
                    type=type.type, optional=optional, roles=roles, stages=stages
                ),
            )
        else:
            raise ValueError(f"bad type: {type}")

        def _wrap(func):
            return func

        return _wrap

    def get_spec(self):
        input_artifacts = {}
        output_artifacts = {}
        for name, (is_output, artifact) in self._artifacts.items():
            if is_output:
                output_artifacts[name] = artifact
            else:
                input_artifacts[name] = artifact
        input_definition = InputDefinitionsSpec(
            parameters=self._params, artifacts=input_artifacts
        )
        output_definition = OutputDefinitionsSpec(artifacts=output_artifacts)
        component = ComponentSpec(
            name=self.name,
            description=self.description,
            provider=self.provider,
            version=self.version,
            labels=[],
            roles=self.roles,
            inputDefinitions=input_definition,
            outputDefinitions=output_definition,
        )
        return ComponentSpecV1(component=component)
