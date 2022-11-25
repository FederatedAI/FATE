from fate.components import GUEST, HOST, DatasetArtifact, Input, Output, Role, cpn


@cpn.component(roles=[GUEST, HOST], provider="fate")
@cpn.artifact("input_data", type=Input[DatasetArtifact], roles=[GUEST, HOST])
@cpn.parameter("method", type=str, default="raw", optional=True)
@cpn.artifact("output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def intersection(
    ctx,
    role: Role,
    input_data,
    method,
    output_data,
):
    if role.is_guest:
        if method == "raw":
            raw_intersect_guest(ctx, input_data, output_data)
    elif role.is_host:
        if method == "raw":
            raw_intersect_host(ctx, input_data, output_data)


def raw_intersect_guest(ctx, input_data, output_data):
    from fate.ml.intersection import RawIntersectionGuest

    data = ctx.reader(input_data).read_dataframe().data
    guest_intersect_obj = RawIntersectionGuest()
    intersect_data = guest_intersect_obj.fit(ctx, data)
    ctx.writer(output_data).write_dataframe(intersect_data)


def raw_intersect_host(ctx, input_data, output_data):
    from fate.ml.intersection import RawIntersectionHost

    data = ctx.reader(input_data).read_dataframe().data
    host_intersect_obj = RawIntersectionHost()
    intersect_data = host_intersect_obj.fit(ctx, data)
    ctx.writer(output_data).write_dataframe(intersect_data)
