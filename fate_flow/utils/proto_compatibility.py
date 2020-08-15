from fate_arch.common import log

try:
    from eggroll.core.proto import basic_meta_pb2
    from eggroll.core.proto import proxy_pb2, proxy_pb2_grpc
except ImportError as e:
    # LOGGER.warning("can't import pb from eggroll 2.x, assuming eggroll 1.x used")
    from fate_arch.protobuf.python import fate_meta_pb2 as basic_meta_pb2
    from fate_arch.protobuf.python import fate_proxy_pb2 as proxy_pb2
    from fate_arch.protobuf.python import fate_proxy_pb2_grpc as proxy_pb2_grpc
