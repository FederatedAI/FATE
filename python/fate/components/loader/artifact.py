def load_artifact(data, artifact_type):
    from fate.components import (
        Artifact,
        Artifacts,
        DatasetArtifact,
        DatasetArtifacts,
        MetricArtifact,
        ModelArtifact,
        ModelArtifacts,
    )

    if isinstance(data, list):
        if artifact_type == DatasetArtifacts:
            return DatasetArtifacts([DatasetArtifact(name=d.name, uri=d.uri, metadata=d.metadata) for d in data])
        if artifact_type == ModelArtifacts:
            return ModelArtifacts([ModelArtifact(name=d.name, uri=d.uri, metadata=d.metadata) for d in data])
        return Artifacts([Artifact(name=d.name, uri=d.uri, metadata=d.metadata) for d in data])
    else:
        if artifact_type == DatasetArtifact:
            return DatasetArtifact(name=data.name, uri=data.uri, metadata=data.metadata)
        if artifact_type == ModelArtifact:
            return ModelArtifact(name=data.name, uri=data.uri, metadata=data.metadata)
        if artifact_type == MetricArtifact:
            return MetricArtifact(name=data.name, uri=data.uri, metadata=data.metadata)
        return Artifact(name=data.name, uri=data.uri, metadata=data.metadata)
