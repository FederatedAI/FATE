from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()

try:
    from eggroll.core.proto import basic_meta_pb2
    from eggroll.core.proto import proxy_pb2, proxy_pb2_grpc
    from eggroll.core.proto import proxy_pb2_grpc
except ImportError as e:
    LOGGER.warning("can't import pb from eggroll 2.x, assuming eggroll 1.x used")

    try:
        from eggroll.api.proto import basic_meta_pb2
        from eggroll.api.proto import proxy_pb2, proxy_pb2_grpc
        from eggroll.api.proto import proxy_pb2_grpc
    except ImportError as e:
        LOGGER.warning("eggroll 1.x not deployed, assuming use embed standalone version")
        from arch.standalone.proto import basic_meta_pb2
        from arch.standalone.proto import proxy_pb2, proxy_pb2_grpc
        from arch.standalone.proto import proxy_pb2_grpc
