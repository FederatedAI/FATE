package com.webank.ai.eggroll.api.driver.clustercomm;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * submit transfer job
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/cluster-comm.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class TransferSubmitServiceGrpc {

  private TransferSubmitServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.eggroll.api.driver.clustercomm.TransferSubmitService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getSendMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "send",
      requestType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      responseType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getSendMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getSendMethod;
    if ((getSendMethod = TransferSubmitServiceGrpc.getSendMethod) == null) {
      synchronized (TransferSubmitServiceGrpc.class) {
        if ((getSendMethod = TransferSubmitServiceGrpc.getSendMethod) == null) {
          TransferSubmitServiceGrpc.getSendMethod = getSendMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "send"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setSchemaDescriptor(new TransferSubmitServiceMethodDescriptorSupplier("send"))
              .build();
        }
      }
    }
    return getSendMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getRecvMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "recv",
      requestType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      responseType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getRecvMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getRecvMethod;
    if ((getRecvMethod = TransferSubmitServiceGrpc.getRecvMethod) == null) {
      synchronized (TransferSubmitServiceGrpc.class) {
        if ((getRecvMethod = TransferSubmitServiceGrpc.getRecvMethod) == null) {
          TransferSubmitServiceGrpc.getRecvMethod = getRecvMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "recv"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setSchemaDescriptor(new TransferSubmitServiceMethodDescriptorSupplier("recv"))
              .build();
        }
      }
    }
    return getRecvMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getCheckStatusNowMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "checkStatusNow",
      requestType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      responseType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getCheckStatusNowMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getCheckStatusNowMethod;
    if ((getCheckStatusNowMethod = TransferSubmitServiceGrpc.getCheckStatusNowMethod) == null) {
      synchronized (TransferSubmitServiceGrpc.class) {
        if ((getCheckStatusNowMethod = TransferSubmitServiceGrpc.getCheckStatusNowMethod) == null) {
          TransferSubmitServiceGrpc.getCheckStatusNowMethod = getCheckStatusNowMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "checkStatusNow"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setSchemaDescriptor(new TransferSubmitServiceMethodDescriptorSupplier("checkStatusNow"))
              .build();
        }
      }
    }
    return getCheckStatusNowMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getCheckStatusMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "checkStatus",
      requestType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      responseType = com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
      com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getCheckStatusMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> getCheckStatusMethod;
    if ((getCheckStatusMethod = TransferSubmitServiceGrpc.getCheckStatusMethod) == null) {
      synchronized (TransferSubmitServiceGrpc.class) {
        if ((getCheckStatusMethod = TransferSubmitServiceGrpc.getCheckStatusMethod) == null) {
          TransferSubmitServiceGrpc.getCheckStatusMethod = getCheckStatusMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta, com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "checkStatus"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta.getDefaultInstance()))
              .setSchemaDescriptor(new TransferSubmitServiceMethodDescriptorSupplier("checkStatus"))
              .build();
        }
      }
    }
    return getCheckStatusMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static TransferSubmitServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TransferSubmitServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TransferSubmitServiceStub>() {
        @java.lang.Override
        public TransferSubmitServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TransferSubmitServiceStub(channel, callOptions);
        }
      };
    return TransferSubmitServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static TransferSubmitServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TransferSubmitServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TransferSubmitServiceBlockingStub>() {
        @java.lang.Override
        public TransferSubmitServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TransferSubmitServiceBlockingStub(channel, callOptions);
        }
      };
    return TransferSubmitServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static TransferSubmitServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TransferSubmitServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TransferSubmitServiceFutureStub>() {
        @java.lang.Override
        public TransferSubmitServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TransferSubmitServiceFutureStub(channel, callOptions);
        }
      };
    return TransferSubmitServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * submit transfer job
   * </pre>
   */
  public interface AsyncService {

    /**
     * <pre>
     * send data
     * </pre>
     */
    default void send(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSendMethod(), responseObserver);
    }

    /**
     * <pre>
     * receive data, i.e. wait for data to arrive
     * </pre>
     */
    default void recv(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRecvMethod(), responseObserver);
    }

    /**
     * <pre>
     * check the transfer status, return immediately
     * </pre>
     */
    default void checkStatusNow(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCheckStatusNowMethod(), responseObserver);
    }

    /**
     * <pre>
     * check the transfer status, block until finished or default timeout
     * </pre>
     */
    default void checkStatus(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCheckStatusMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service TransferSubmitService.
   * <pre>
   * submit transfer job
   * </pre>
   */
  public static abstract class TransferSubmitServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return TransferSubmitServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service TransferSubmitService.
   * <pre>
   * submit transfer job
   * </pre>
   */
  public static final class TransferSubmitServiceStub
      extends io.grpc.stub.AbstractAsyncStub<TransferSubmitServiceStub> {
    private TransferSubmitServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TransferSubmitServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TransferSubmitServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     * send data
     * </pre>
     */
    public void send(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSendMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * receive data, i.e. wait for data to arrive
     * </pre>
     */
    public void recv(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRecvMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * check the transfer status, return immediately
     * </pre>
     */
    public void checkStatusNow(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCheckStatusNowMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * check the transfer status, block until finished or default timeout
     * </pre>
     */
    public void checkStatus(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCheckStatusMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service TransferSubmitService.
   * <pre>
   * submit transfer job
   * </pre>
   */
  public static final class TransferSubmitServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<TransferSubmitServiceBlockingStub> {
    private TransferSubmitServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TransferSubmitServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TransferSubmitServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     * send data
     * </pre>
     */
    public com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta send(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSendMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * receive data, i.e. wait for data to arrive
     * </pre>
     */
    public com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta recv(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRecvMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * check the transfer status, return immediately
     * </pre>
     */
    public com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta checkStatusNow(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCheckStatusNowMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * check the transfer status, block until finished or default timeout
     * </pre>
     */
    public com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta checkStatus(com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCheckStatusMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service TransferSubmitService.
   * <pre>
   * submit transfer job
   * </pre>
   */
  public static final class TransferSubmitServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<TransferSubmitServiceFutureStub> {
    private TransferSubmitServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TransferSubmitServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TransferSubmitServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     * send data
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> send(
        com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSendMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * receive data, i.e. wait for data to arrive
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> recv(
        com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRecvMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * check the transfer status, return immediately
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> checkStatusNow(
        com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCheckStatusNowMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * check the transfer status, block until finished or default timeout
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta> checkStatus(
        com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCheckStatusMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_SEND = 0;
  private static final int METHODID_RECV = 1;
  private static final int METHODID_CHECK_STATUS_NOW = 2;
  private static final int METHODID_CHECK_STATUS = 3;

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
        case METHODID_SEND:
          serviceImpl.send((com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>) responseObserver);
          break;
        case METHODID_RECV:
          serviceImpl.recv((com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>) responseObserver);
          break;
        case METHODID_CHECK_STATUS_NOW:
          serviceImpl.checkStatusNow((com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>) responseObserver);
          break;
        case METHODID_CHECK_STATUS:
          serviceImpl.checkStatus((com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>) responseObserver);
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
          getSendMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>(
                service, METHODID_SEND)))
        .addMethod(
          getRecvMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>(
                service, METHODID_RECV)))
        .addMethod(
          getCheckStatusNowMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>(
                service, METHODID_CHECK_STATUS_NOW)))
        .addMethod(
          getCheckStatusMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta,
              com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.TransferMeta>(
                service, METHODID_CHECK_STATUS)))
        .build();
  }

  private static abstract class TransferSubmitServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    TransferSubmitServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("TransferSubmitService");
    }
  }

  private static final class TransferSubmitServiceFileDescriptorSupplier
      extends TransferSubmitServiceBaseDescriptorSupplier {
    TransferSubmitServiceFileDescriptorSupplier() {}
  }

  private static final class TransferSubmitServiceMethodDescriptorSupplier
      extends TransferSubmitServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    TransferSubmitServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (TransferSubmitServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new TransferSubmitServiceFileDescriptorSupplier())
              .addMethod(getSendMethod())
              .addMethod(getRecvMethod())
              .addMethod(getCheckStatusNowMethod())
              .addMethod(getCheckStatusMethod())
              .build();
        }
      }
    }
    return result;
  }
}
