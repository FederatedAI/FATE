from arch.api.utils import log_utils

try:
    from eggroll.core.proto import basic_meta_pb2
    from eggroll.core.proto import proxy_pb2, proxy_pb2_grpc
except ImportError as e:
    # LOGGER.warning("can't import pb from eggroll 2.x, assuming eggroll 1.x used")
    from arch.api.proto import fate_meta_pb2 as basic_meta_pb2
    from arch.api.proto import fate_proxy_pb2 as proxy_pb2
    from arch.api.proto import fate_proxy_pb2_grpc as proxy_pb2_grpc
