from fate.components import cpn
from fate.components.spec import DatasetArtifact, Input, Output, roles
from fate.ml.intersection import RawIntersectionGuest, RawIntersectionHost


@cpn.component(roles=[roles.GUEST, roles.HOST], provider="fate", version="2.0.0.alpha")
@cpn.artifact("input_data", type=Input[DatasetArtifact], roles=[roles.GUEST, roles.HOST])
@cpn.parameter("method", type=str, default="raw", optional=True)
@cpn.artifact("output_data", type=Output[DatasetArtifact], roles=[roles.GUEST, roles.HOST])
def intersection(
    ctx,
    role,
    input_data,
    method,
    output_data,
):
    if role == "guest":
        if method == "raw":
            raw_intersect_guest(ctx, input_data, output_data)
    elif role == "host":
        if method == "raw":
            raw_intersect_host(ctx, input_data, output_data)


def raw_intersect_guest(ctx, input_data, output_data):
    data = ctx.reader(input_data).read_dataframe().data
    guest_intersect_obj = RawIntersectionGuest()
    intersect_data = guest_intersect_obj.fit(ctx, data)
    ctx.writer(output_data).write_dataframe(intersect_data)


def raw_intersect_host(ctx,  input_data, output_data):
    data = ctx.reader(input_data).read_dataframe().data
    host_intersect_obj = RawIntersectionHost()
    intersect_data = host_intersect_obj.fit(ctx, data)
    ctx.writer(output_data).write_dataframe(intersect_data)
