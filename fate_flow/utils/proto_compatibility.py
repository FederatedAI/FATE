from arch.api import _EGGROLL_VERSION

if _EGGROLL_VERSION < 2:
    from eggroll.api.proto import basic_meta_pb2
    from eggroll.api.proto import proxy_pb2, proxy_pb2_grpc
    from eggroll.api.proto import proxy_pb2_grpc
else:
    from eggroll.core.proto import basic_meta_pb2
    from eggroll.core.proto import proxy_pb2, proxy_pb2_grpc
    from eggroll.core.proto import proxy_pb2_grpc
