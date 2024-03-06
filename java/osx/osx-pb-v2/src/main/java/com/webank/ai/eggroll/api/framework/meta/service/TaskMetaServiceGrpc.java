package com.webank.ai.eggroll.api.framework.meta.service;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * will be implemented in stage 2
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/meta-service.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class TaskMetaServiceGrpc {

  private TaskMetaServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.eggroll.api.framework.meta.service.TaskMetaService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTaskMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "createTask",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTaskMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTaskMethod;
    if ((getCreateTaskMethod = TaskMetaServiceGrpc.getCreateTaskMethod) == null) {
      synchronized (TaskMetaServiceGrpc.class) {
        if ((getCreateTaskMethod = TaskMetaServiceGrpc.getCreateTaskMethod) == null) {
          TaskMetaServiceGrpc.getCreateTaskMethod = getCreateTaskMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "createTask"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new TaskMetaServiceMethodDescriptorSupplier("createTask"))
              .build();
        }
      }
    }
    return getCreateTaskMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateTaskMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "updateTask",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateTaskMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateTaskMethod;
    if ((getUpdateTaskMethod = TaskMetaServiceGrpc.getUpdateTaskMethod) == null) {
      synchronized (TaskMetaServiceGrpc.class) {
        if ((getUpdateTaskMethod = TaskMetaServiceGrpc.getUpdateTaskMethod) == null) {
          TaskMetaServiceGrpc.getUpdateTaskMethod = getUpdateTaskMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "updateTask"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new TaskMetaServiceMethodDescriptorSupplier("updateTask"))
              .build();
        }
      }
    }
    return getUpdateTaskMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateResultMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "createResult",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateResultMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateResultMethod;
    if ((getCreateResultMethod = TaskMetaServiceGrpc.getCreateResultMethod) == null) {
      synchronized (TaskMetaServiceGrpc.class) {
        if ((getCreateResultMethod = TaskMetaServiceGrpc.getCreateResultMethod) == null) {
          TaskMetaServiceGrpc.getCreateResultMethod = getCreateResultMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "createResult"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new TaskMetaServiceMethodDescriptorSupplier("createResult"))
              .build();
        }
      }
    }
    return getCreateResultMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateResultMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "updateResult",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateResultMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateResultMethod;
    if ((getUpdateResultMethod = TaskMetaServiceGrpc.getUpdateResultMethod) == null) {
      synchronized (TaskMetaServiceGrpc.class) {
        if ((getUpdateResultMethod = TaskMetaServiceGrpc.getUpdateResultMethod) == null) {
          TaskMetaServiceGrpc.getUpdateResultMethod = getUpdateResultMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "updateResult"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new TaskMetaServiceMethodDescriptorSupplier("updateResult"))
              .build();
        }
      }
    }
    return getUpdateResultMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetResultByIdMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getResultById",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetResultByIdMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetResultByIdMethod;
    if ((getGetResultByIdMethod = TaskMetaServiceGrpc.getGetResultByIdMethod) == null) {
      synchronized (TaskMetaServiceGrpc.class) {
        if ((getGetResultByIdMethod = TaskMetaServiceGrpc.getGetResultByIdMethod) == null) {
          TaskMetaServiceGrpc.getGetResultByIdMethod = getGetResultByIdMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getResultById"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new TaskMetaServiceMethodDescriptorSupplier("getResultById"))
              .build();
        }
      }
    }
    return getGetResultByIdMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static TaskMetaServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TaskMetaServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TaskMetaServiceStub>() {
        @java.lang.Override
        public TaskMetaServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TaskMetaServiceStub(channel, callOptions);
        }
      };
    return TaskMetaServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static TaskMetaServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TaskMetaServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TaskMetaServiceBlockingStub>() {
        @java.lang.Override
        public TaskMetaServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TaskMetaServiceBlockingStub(channel, callOptions);
        }
      };
    return TaskMetaServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static TaskMetaServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TaskMetaServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TaskMetaServiceFutureStub>() {
        @java.lang.Override
        public TaskMetaServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TaskMetaServiceFutureStub(channel, callOptions);
        }
      };
    return TaskMetaServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * will be implemented in stage 2
   * </pre>
   */
  public interface AsyncService {

    /**
     */
    default void createTask(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCreateTaskMethod(), responseObserver);
    }

    /**
     */
    default void updateTask(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUpdateTaskMethod(), responseObserver);
    }

    /**
     */
    default void createResult(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCreateResultMethod(), responseObserver);
    }

    /**
     */
    default void updateResult(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUpdateResultMethod(), responseObserver);
    }

    /**
     */
    default void getResultById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetResultByIdMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service TaskMetaService.
   * <pre>
   * will be implemented in stage 2
   * </pre>
   */
  public static abstract class TaskMetaServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return TaskMetaServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service TaskMetaService.
   * <pre>
   * will be implemented in stage 2
   * </pre>
   */
  public static final class TaskMetaServiceStub
      extends io.grpc.stub.AbstractAsyncStub<TaskMetaServiceStub> {
    private TaskMetaServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TaskMetaServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TaskMetaServiceStub(channel, callOptions);
    }

    /**
     */
    public void createTask(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCreateTaskMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void updateTask(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUpdateTaskMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void createResult(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCreateResultMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void updateResult(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUpdateResultMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getResultById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetResultByIdMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service TaskMetaService.
   * <pre>
   * will be implemented in stage 2
   * </pre>
   */
  public static final class TaskMetaServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<TaskMetaServiceBlockingStub> {
    private TaskMetaServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TaskMetaServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TaskMetaServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse createTask(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCreateTaskMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse updateTask(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUpdateTaskMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse createResult(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCreateResultMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse updateResult(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUpdateResultMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getResultById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetResultByIdMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service TaskMetaService.
   * <pre>
   * will be implemented in stage 2
   * </pre>
   */
  public static final class TaskMetaServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<TaskMetaServiceFutureStub> {
    private TaskMetaServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TaskMetaServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TaskMetaServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> createTask(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCreateTaskMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> updateTask(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUpdateTaskMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> createResult(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCreateResultMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> updateResult(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUpdateResultMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getResultById(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetResultByIdMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_CREATE_TASK = 0;
  private static final int METHODID_UPDATE_TASK = 1;
  private static final int METHODID_CREATE_RESULT = 2;
  private static final int METHODID_UPDATE_RESULT = 3;
  private static final int METHODID_GET_RESULT_BY_ID = 4;

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
        case METHODID_CREATE_TASK:
          serviceImpl.createTask((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_UPDATE_TASK:
          serviceImpl.updateTask((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_CREATE_RESULT:
          serviceImpl.createResult((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_UPDATE_RESULT:
          serviceImpl.updateResult((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_RESULT_BY_ID:
          serviceImpl.getResultById((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
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
          getCreateTaskMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_CREATE_TASK)))
        .addMethod(
          getUpdateTaskMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_UPDATE_TASK)))
        .addMethod(
          getCreateResultMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_CREATE_RESULT)))
        .addMethod(
          getUpdateResultMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_UPDATE_RESULT)))
        .addMethod(
          getGetResultByIdMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_RESULT_BY_ID)))
        .build();
  }

  private static abstract class TaskMetaServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    TaskMetaServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.eggroll.api.framework.meta.service.MetaService.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("TaskMetaService");
    }
  }

  private static final class TaskMetaServiceFileDescriptorSupplier
      extends TaskMetaServiceBaseDescriptorSupplier {
    TaskMetaServiceFileDescriptorSupplier() {}
  }

  private static final class TaskMetaServiceMethodDescriptorSupplier
      extends TaskMetaServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    TaskMetaServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (TaskMetaServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new TaskMetaServiceFileDescriptorSupplier())
              .addMethod(getCreateTaskMethod())
              .addMethod(getUpdateTaskMethod())
              .addMethod(getCreateResultMethod())
              .addMethod(getUpdateResultMethod())
              .addMethod(getGetResultByIdMethod())
              .build();
        }
      }
    }
    return result;
  }
}
