import inspect
from typing import List, Optional

from fate.components.core.essential import Role

from ._json import JsonMetricArtifactDescribe


def json_metric_output(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_metric_artifact(_create_json_metric_artifact_describe(name, roles, desc, optional, multi=False))


def json_metric_outputs(name: str, roles: Optional[List[Role]] = None, desc="", optional=False):
    return _output_metric_artifact(_create_json_metric_artifact_describe(name, roles, desc, optional, multi=True))


def _output_metric_artifact(desc):
    def decorator(f):
        if not hasattr(f, "__component_artifacts__"):
            from ..._component_artifact import ComponentArtifactDescribes

            f.__component_artifacts__ = ComponentArtifactDescribes()

        f.__component_artifacts__.add_metric_output(desc)
        return f

    return decorator


def _prepare(roles, desc):
    if roles is None:
        roles = []
    if desc:
        desc = inspect.cleandoc(desc)
    return roles, desc


def _create_json_metric_artifact_describe(name, roles, desc, optional, multi):
    roles, desc = _prepare(roles, desc)
    return JsonMetricArtifactDescribe(name=name, roles=roles, stages=[], desc=desc, optional=optional, multi=multi)
