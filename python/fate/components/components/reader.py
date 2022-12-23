from fate.arch.unify import URI
from fate.components import GUEST, HOST, DatasetArtifact, Output, Role, cpn


@cpn.component(roles=[GUEST, HOST])
@cpn.parameter("path", type=str, default=None, optional=False)
@cpn.parameter("format", type=str, default="csv", optional=False)
@cpn.parameter("id_name", type=str, default="id", optional=True)
@cpn.parameter("delimiter", type=str, default=",", optional=True)
@cpn.parameter("label_name", type=str, default=None, optional=True)
@cpn.parameter("label_type", type=str, default="float32", optional=True)
@cpn.parameter("dtype", type=str, default="float32", optional=True)
@cpn.artifact("output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def reader(
    ctx,
    role: Role,
    path,
    format,
    id_name,
    delimiter,
    label_name,
    label_type,
    dtype,
    output_data,
):
    read_data(ctx, path, format, id_name, delimiter, label_name, label_type, dtype, output_data)


def read_data(ctx, path, format, id_name, delimiter, label_name, label_type, dtype, output_data):
    if format == "csv":
        data_meta = DatasetArtifact(
            uri=path,
            name="data",
            metadata=dict(
                format=format,
                id_name=id_name,
                delimiter=delimiter,
                label_name=label_name,
                label_type=label_type,
                dtype=dtype,
            ),
        )
        data = ctx.reader(data_meta).read_dataframe()
        ctx.writer(output_data).write_dataframe(data)
