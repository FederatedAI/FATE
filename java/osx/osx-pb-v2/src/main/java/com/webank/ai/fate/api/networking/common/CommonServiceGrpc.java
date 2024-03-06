package com.webank.ai.fate.api.networking.common;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/common-service.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class CommonServiceGrpc {

  private CommonServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.fate.api.networking.common.CommonService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getQueryMetricsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "queryMetrics",
      requestType = com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest.class,
      responseType = com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getQueryMetricsMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getQueryMetricsMethod;
    if ((getQueryMetricsMethod = CommonServiceGrpc.getQueryMetricsMethod) == null) {
      synchronized (CommonServiceGrpc.class) {
        if ((getQueryMetricsMethod = CommonServiceGrpc.getQueryMetricsMethod) == null) {
          CommonServiceGrpc.getQueryMetricsMethod = getQueryMetricsMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "queryMetrics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CommonServiceMethodDescriptorSupplier("queryMetrics"))
              .build();
        }
      }
    }
    return getQueryMetricsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getUpdateFlowRuleMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "updateFlowRule",
      requestType = com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest.class,
      responseType = com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getUpdateFlowRuleMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getUpdateFlowRuleMethod;
    if ((getUpdateFlowRuleMethod = CommonServiceGrpc.getUpdateFlowRuleMethod) == null) {
      synchronized (CommonServiceGrpc.class) {
        if ((getUpdateFlowRuleMethod = CommonServiceGrpc.getUpdateFlowRuleMethod) == null) {
          CommonServiceGrpc.getUpdateFlowRuleMethod = getUpdateFlowRuleMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "updateFlowRule"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CommonServiceMethodDescriptorSupplier("updateFlowRule"))
              .build();
        }
      }
    }
    return getUpdateFlowRuleMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getListPropsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "listProps",
      requestType = com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest.class,
      responseType = com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getListPropsMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getListPropsMethod;
    if ((getListPropsMethod = CommonServiceGrpc.getListPropsMethod) == null) {
      synchronized (CommonServiceGrpc.class) {
        if ((getListPropsMethod = CommonServiceGrpc.getListPropsMethod) == null) {
          CommonServiceGrpc.getListPropsMethod = getListPropsMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "listProps"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CommonServiceMethodDescriptorSupplier("listProps"))
              .build();
        }
      }
    }
    return getListPropsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getQueryJvmInfoMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "queryJvmInfo",
      requestType = com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest.class,
      responseType = com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest,
      com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getQueryJvmInfoMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> getQueryJvmInfoMethod;
    if ((getQueryJvmInfoMethod = CommonServiceGrpc.getQueryJvmInfoMethod) == null) {
      synchronized (CommonServiceGrpc.class) {
        if ((getQueryJvmInfoMethod = CommonServiceGrpc.getQueryJvmInfoMethod) == null) {
          CommonServiceGrpc.getQueryJvmInfoMethod = getQueryJvmInfoMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest, com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "queryJvmInfo"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CommonServiceMethodDescriptorSupplier("queryJvmInfo"))
              .build();
        }
      }
    }
    return getQueryJvmInfoMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static CommonServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CommonServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CommonServiceStub>() {
        @java.lang.Override
        public CommonServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CommonServiceStub(channel, callOptions);
        }
      };
    return CommonServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static CommonServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CommonServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CommonServiceBlockingStub>() {
        @java.lang.Override
        public CommonServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CommonServiceBlockingStub(channel, callOptions);
        }
      };
    return CommonServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static CommonServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CommonServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CommonServiceFutureStub>() {
        @java.lang.Override
        public CommonServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CommonServiceFutureStub(channel, callOptions);
        }
      };
    return CommonServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void queryMetrics(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getQueryMetricsMethod(), responseObserver);
    }

    /**
     */
    default void updateFlowRule(com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUpdateFlowRuleMethod(), responseObserver);
    }

    /**
     */
    default void listProps(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getListPropsMethod(), responseObserver);
    }

    /**
     */
    default void queryJvmInfo(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getQueryJvmInfoMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service CommonService.
   */
  public static abstract class CommonServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return CommonServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service CommonService.
   */
  public static final class CommonServiceStub
      extends io.grpc.stub.AbstractAsyncStub<CommonServiceStub> {
    private CommonServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CommonServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CommonServiceStub(channel, callOptions);
    }

    /**
     */
    public void queryMetrics(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getQueryMetricsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void updateFlowRule(com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUpdateFlowRuleMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void listProps(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getListPropsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void queryJvmInfo(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getQueryJvmInfoMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service CommonService.
   */
  public static final class CommonServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<CommonServiceBlockingStub> {
    private CommonServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CommonServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CommonServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse queryMetrics(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getQueryMetricsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse updateFlowRule(com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUpdateFlowRuleMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse listProps(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getListPropsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse queryJvmInfo(com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getQueryJvmInfoMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service CommonService.
   */
  public static final class CommonServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<CommonServiceFutureStub> {
    private CommonServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CommonServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CommonServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> queryMetrics(
        com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getQueryMetricsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> updateFlowRule(
        com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUpdateFlowRuleMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> listProps(
        com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getListPropsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse> queryJvmInfo(
        com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getQueryJvmInfoMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_QUERY_METRICS = 0;
  private static final int METHODID_UPDATE_FLOW_RULE = 1;
  private static final int METHODID_LIST_PROPS = 2;
  private static final int METHODID_QUERY_JVM_INFO = 3;

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
        case METHODID_QUERY_METRICS:
          serviceImpl.queryMetrics((com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>) responseObserver);
          break;
        case METHODID_UPDATE_FLOW_RULE:
          serviceImpl.updateFlowRule((com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>) responseObserver);
          break;
        case METHODID_LIST_PROPS:
          serviceImpl.listProps((com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>) responseObserver);
          break;
        case METHODID_QUERY_JVM_INFO:
          serviceImpl.queryJvmInfo((com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>) responseObserver);
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
          getQueryMetricsMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryMetricRequest,
              com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>(
                service, METHODID_QUERY_METRICS)))
        .addMethod(
          getUpdateFlowRuleMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.fate.api.networking.common.CommonServiceProto.UpdateFlowRuleRequest,
              com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>(
                service, METHODID_UPDATE_FLOW_RULE)))
        .addMethod(
          getListPropsMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryPropsRequest,
              com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>(
                service, METHODID_LIST_PROPS)))
        .addMethod(
          getQueryJvmInfoMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.fate.api.networking.common.CommonServiceProto.QueryJvmInfoRequest,
              com.webank.ai.fate.api.networking.common.CommonServiceProto.CommonResponse>(
                service, METHODID_QUERY_JVM_INFO)))
        .build();
  }

  private static abstract class CommonServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    CommonServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.fate.api.networking.common.CommonServiceProto.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("CommonService");
    }
  }

  private static final class CommonServiceFileDescriptorSupplier
      extends CommonServiceBaseDescriptorSupplier {
    CommonServiceFileDescriptorSupplier() {}
  }

  private static final class CommonServiceMethodDescriptorSupplier
      extends CommonServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    CommonServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (CommonServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new CommonServiceFileDescriptorSupplier())
              .addMethod(getQueryMetricsMethod())
              .addMethod(getUpdateFlowRuleMethod())
              .addMethod(getListPropsMethod())
              .addMethod(getQueryJvmInfoMethod())
              .build();
        }
      }
    }
    return result;
  }
}
