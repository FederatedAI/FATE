package com.webank.ai.eggroll.api.networking.proxy;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * data transfer service
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/proxy.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class DataTransferServiceGrpc {

  private DataTransferServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.eggroll.api.networking.proxy.DataTransferService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> getPushMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "push",
      requestType = com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.class,
      responseType = com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata.class,
      methodType = io.grpc.MethodDescriptor.MethodType.CLIENT_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> getPushMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet, com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> getPushMethod;
    if ((getPushMethod = DataTransferServiceGrpc.getPushMethod) == null) {
      synchronized (DataTransferServiceGrpc.class) {
        if ((getPushMethod = DataTransferServiceGrpc.getPushMethod) == null) {
          DataTransferServiceGrpc.getPushMethod = getPushMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet, com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.CLIENT_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "push"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata.getDefaultInstance()))
              .setSchemaDescriptor(new DataTransferServiceMethodDescriptorSupplier("push"))
              .build();
        }
      }
    }
    return getPushMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> getPullMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "pull",
      requestType = com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata.class,
      responseType = com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.class,
      methodType = io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> getPullMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata, com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> getPullMethod;
    if ((getPullMethod = DataTransferServiceGrpc.getPullMethod) == null) {
      synchronized (DataTransferServiceGrpc.class) {
        if ((getPullMethod = DataTransferServiceGrpc.getPullMethod) == null) {
          DataTransferServiceGrpc.getPullMethod = getPullMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata, com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "pull"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.getDefaultInstance()))
              .setSchemaDescriptor(new DataTransferServiceMethodDescriptorSupplier("pull"))
              .build();
        }
      }
    }
    return getPullMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> getUnaryCallMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "unaryCall",
      requestType = com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.class,
      responseType = com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> getUnaryCallMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet, com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> getUnaryCallMethod;
    if ((getUnaryCallMethod = DataTransferServiceGrpc.getUnaryCallMethod) == null) {
      synchronized (DataTransferServiceGrpc.class) {
        if ((getUnaryCallMethod = DataTransferServiceGrpc.getUnaryCallMethod) == null) {
          DataTransferServiceGrpc.getUnaryCallMethod = getUnaryCallMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet, com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "unaryCall"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet.getDefaultInstance()))
              .setSchemaDescriptor(new DataTransferServiceMethodDescriptorSupplier("unaryCall"))
              .build();
        }
      }
    }
    return getUnaryCallMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> getPollingMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "polling",
      requestType = com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame.class,
      responseType = com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame.class,
      methodType = io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame,
      com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> getPollingMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame, com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> getPollingMethod;
    if ((getPollingMethod = DataTransferServiceGrpc.getPollingMethod) == null) {
      synchronized (DataTransferServiceGrpc.class) {
        if ((getPollingMethod = DataTransferServiceGrpc.getPollingMethod) == null) {
          DataTransferServiceGrpc.getPollingMethod = getPollingMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame, com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "polling"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame.getDefaultInstance()))
              .setSchemaDescriptor(new DataTransferServiceMethodDescriptorSupplier("polling"))
              .build();
        }
      }
    }
    return getPollingMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static DataTransferServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<DataTransferServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<DataTransferServiceStub>() {
        @java.lang.Override
        public DataTransferServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new DataTransferServiceStub(channel, callOptions);
        }
      };
    return DataTransferServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static DataTransferServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<DataTransferServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<DataTransferServiceBlockingStub>() {
        @java.lang.Override
        public DataTransferServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new DataTransferServiceBlockingStub(channel, callOptions);
        }
      };
    return DataTransferServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static DataTransferServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<DataTransferServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<DataTransferServiceFutureStub>() {
        @java.lang.Override
        public DataTransferServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new DataTransferServiceFutureStub(channel, callOptions);
        }
      };
    return DataTransferServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * data transfer service
   * </pre>
   */
  public interface AsyncService {

    /**
     */
    default io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> push(
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getPushMethod(), responseObserver);
    }

    /**
     */
    default void pull(com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPullMethod(), responseObserver);
    }

    /**
     */
    default void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUnaryCallMethod(), responseObserver);
    }

    /**
     */
    default io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> polling(
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getPollingMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service DataTransferService.
   * <pre>
   * data transfer service
   * </pre>
   */
  public static abstract class DataTransferServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return DataTransferServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service DataTransferService.
   * <pre>
   * data transfer service
   * </pre>
   */
  public static final class DataTransferServiceStub
      extends io.grpc.stub.AbstractAsyncStub<DataTransferServiceStub> {
    private DataTransferServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected DataTransferServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new DataTransferServiceStub(channel, callOptions);
    }

    /**
     */
    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> push(
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncClientStreamingCall(
          getChannel().newCall(getPushMethod(), getCallOptions()), responseObserver);
    }

    /**
     */
    public void pull(com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
      io.grpc.stub.ClientCalls.asyncServerStreamingCall(
          getChannel().newCall(getPullMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUnaryCallMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> polling(
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncBidiStreamingCall(
          getChannel().newCall(getPollingMethod(), getCallOptions()), responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service DataTransferService.
   * <pre>
   * data transfer service
   * </pre>
   */
  public static final class DataTransferServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<DataTransferServiceBlockingStub> {
    private DataTransferServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected DataTransferServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new DataTransferServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public java.util.Iterator<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> pull(
        com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata request) {
      return io.grpc.stub.ClientCalls.blockingServerStreamingCall(
          getChannel(), getPullMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUnaryCallMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service DataTransferService.
   * <pre>
   * data transfer service
   * </pre>
   */
  public static final class DataTransferServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<DataTransferServiceFutureStub> {
    private DataTransferServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected DataTransferServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new DataTransferServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> unaryCall(
        com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUnaryCallMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_PULL = 0;
  private static final int METHODID_UNARY_CALL = 1;
  private static final int METHODID_PUSH = 2;
  private static final int METHODID_POLLING = 3;

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
        case METHODID_PULL:
          serviceImpl.pull((com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet>) responseObserver);
          break;
        case METHODID_UNARY_CALL:
          serviceImpl.unaryCall((com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet>) responseObserver);
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
        case METHODID_PUSH:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.push(
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata>) responseObserver);
        case METHODID_POLLING:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.polling(
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame>) responseObserver);
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getPushMethod(),
          io.grpc.stub.ServerCalls.asyncClientStreamingCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet,
              com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata>(
                service, METHODID_PUSH)))
        .addMethod(
          getPullMethod(),
          io.grpc.stub.ServerCalls.asyncServerStreamingCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata,
              com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet>(
                service, METHODID_PULL)))
        .addMethod(
          getUnaryCallMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet,
              com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet>(
                service, METHODID_UNARY_CALL)))
        .addMethod(
          getPollingMethod(),
          io.grpc.stub.ServerCalls.asyncBidiStreamingCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame,
              com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame>(
                service, METHODID_POLLING)))
        .build();
  }

  private static abstract class DataTransferServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    DataTransferServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.eggroll.api.networking.proxy.Proxy.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("DataTransferService");
    }
  }

  private static final class DataTransferServiceFileDescriptorSupplier
      extends DataTransferServiceBaseDescriptorSupplier {
    DataTransferServiceFileDescriptorSupplier() {}
  }

  private static final class DataTransferServiceMethodDescriptorSupplier
      extends DataTransferServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    DataTransferServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (DataTransferServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new DataTransferServiceFileDescriptorSupplier())
              .addMethod(getPushMethod())
              .addMethod(getPullMethod())
              .addMethod(getUnaryCallMethod())
              .addMethod(getPollingMethod())
              .build();
        }
      }
    }
    return result;
  }
}
