package com.webank.ai.eggroll.api.framework.meta.service;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * service to change node status
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/meta-service.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class ClusterMetaServiceGrpc {

  private ClusterMetaServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.eggroll.api.framework.meta.service.ClusterMetaService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getRegisterNodeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "registerNode",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getRegisterNodeMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getRegisterNodeMethod;
    if ((getRegisterNodeMethod = ClusterMetaServiceGrpc.getRegisterNodeMethod) == null) {
      synchronized (ClusterMetaServiceGrpc.class) {
        if ((getRegisterNodeMethod = ClusterMetaServiceGrpc.getRegisterNodeMethod) == null) {
          ClusterMetaServiceGrpc.getRegisterNodeMethod = getRegisterNodeMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "registerNode"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ClusterMetaServiceMethodDescriptorSupplier("registerNode"))
              .build();
        }
      }
    }
    return getRegisterNodeMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getDeregisterNodeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "deregisterNode",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getDeregisterNodeMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getDeregisterNodeMethod;
    if ((getDeregisterNodeMethod = ClusterMetaServiceGrpc.getDeregisterNodeMethod) == null) {
      synchronized (ClusterMetaServiceGrpc.class) {
        if ((getDeregisterNodeMethod = ClusterMetaServiceGrpc.getDeregisterNodeMethod) == null) {
          ClusterMetaServiceGrpc.getDeregisterNodeMethod = getDeregisterNodeMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "deregisterNode"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ClusterMetaServiceMethodDescriptorSupplier("deregisterNode"))
              .build();
        }
      }
    }
    return getDeregisterNodeMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getHeartbeatMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "heartbeat",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getHeartbeatMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getHeartbeatMethod;
    if ((getHeartbeatMethod = ClusterMetaServiceGrpc.getHeartbeatMethod) == null) {
      synchronized (ClusterMetaServiceGrpc.class) {
        if ((getHeartbeatMethod = ClusterMetaServiceGrpc.getHeartbeatMethod) == null) {
          ClusterMetaServiceGrpc.getHeartbeatMethod = getHeartbeatMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "heartbeat"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new ClusterMetaServiceMethodDescriptorSupplier("heartbeat"))
              .build();
        }
      }
    }
    return getHeartbeatMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static ClusterMetaServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ClusterMetaServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ClusterMetaServiceStub>() {
        @java.lang.Override
        public ClusterMetaServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ClusterMetaServiceStub(channel, callOptions);
        }
      };
    return ClusterMetaServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static ClusterMetaServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ClusterMetaServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ClusterMetaServiceBlockingStub>() {
        @java.lang.Override
        public ClusterMetaServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ClusterMetaServiceBlockingStub(channel, callOptions);
        }
      };
    return ClusterMetaServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static ClusterMetaServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ClusterMetaServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ClusterMetaServiceFutureStub>() {
        @java.lang.Override
        public ClusterMetaServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ClusterMetaServiceFutureStub(channel, callOptions);
        }
      };
    return ClusterMetaServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * service to change node status
   * </pre>
   */
  public interface AsyncService {

    /**
     */
    default void registerNode(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRegisterNodeMethod(), responseObserver);
    }

    /**
     */
    default void deregisterNode(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDeregisterNodeMethod(), responseObserver);
    }

    /**
     */
    default void heartbeat(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getHeartbeatMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service ClusterMetaService.
   * <pre>
   * service to change node status
   * </pre>
   */
  public static abstract class ClusterMetaServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return ClusterMetaServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service ClusterMetaService.
   * <pre>
   * service to change node status
   * </pre>
   */
  public static final class ClusterMetaServiceStub
      extends io.grpc.stub.AbstractAsyncStub<ClusterMetaServiceStub> {
    private ClusterMetaServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ClusterMetaServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ClusterMetaServiceStub(channel, callOptions);
    }

    /**
     */
    public void registerNode(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRegisterNodeMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void deregisterNode(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDeregisterNodeMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void heartbeat(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getHeartbeatMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service ClusterMetaService.
   * <pre>
   * service to change node status
   * </pre>
   */
  public static final class ClusterMetaServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<ClusterMetaServiceBlockingStub> {
    private ClusterMetaServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ClusterMetaServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ClusterMetaServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse registerNode(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRegisterNodeMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse deregisterNode(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDeregisterNodeMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse heartbeat(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getHeartbeatMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service ClusterMetaService.
   * <pre>
   * service to change node status
   * </pre>
   */
  public static final class ClusterMetaServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<ClusterMetaServiceFutureStub> {
    private ClusterMetaServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ClusterMetaServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ClusterMetaServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> registerNode(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRegisterNodeMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> deregisterNode(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDeregisterNodeMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> heartbeat(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getHeartbeatMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_REGISTER_NODE = 0;
  private static final int METHODID_DEREGISTER_NODE = 1;
  private static final int METHODID_HEARTBEAT = 2;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final AsyncService serviceImpl;
    private final int methodId;

    MethodHandlers(AsyncService serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_REGISTER_NODE:
          serviceImpl.registerNode((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_DEREGISTER_NODE:
          serviceImpl.deregisterNode((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_HEARTBEAT:
          serviceImpl.heartbeat((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getRegisterNodeMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_REGISTER_NODE)))
        .addMethod(
          getDeregisterNodeMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_DEREGISTER_NODE)))
        .addMethod(
          getHeartbeatMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_HEARTBEAT)))
        .build();
  }

  private static abstract class ClusterMetaServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    ClusterMetaServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.eggroll.api.framework.meta.service.MetaService.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("ClusterMetaService");
    }
  }

  private static final class ClusterMetaServiceFileDescriptorSupplier
      extends ClusterMetaServiceBaseDescriptorSupplier {
    ClusterMetaServiceFileDescriptorSupplier() {}
  }

  private static final class ClusterMetaServiceMethodDescriptorSupplier
      extends ClusterMetaServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    ClusterMetaServiceMethodDescriptorSupplier(java.lang.String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (ClusterMetaServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new ClusterMetaServiceFileDescriptorSupplier())
              .addMethod(getRegisterNodeMethod())
              .addMethod(getDeregisterNodeMethod())
              .addMethod(getHeartbeatMethod())
              .build();
        }
      }
    }
    return result;
  }
}
