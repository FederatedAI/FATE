package com.webank.eggroll.core.transfer;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * TODO: use transfer lib
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/transfer.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class TransferServiceGrpc {

  private TransferServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.eggroll.core.transfer.TransferService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch,
      com.webank.eggroll.core.transfer.Transfer.TransferBatch> getSendMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "send",
      requestType = com.webank.eggroll.core.transfer.Transfer.TransferBatch.class,
      responseType = com.webank.eggroll.core.transfer.Transfer.TransferBatch.class,
      methodType = io.grpc.MethodDescriptor.MethodType.CLIENT_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch,
      com.webank.eggroll.core.transfer.Transfer.TransferBatch> getSendMethod() {
    io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch, com.webank.eggroll.core.transfer.Transfer.TransferBatch> getSendMethod;
    if ((getSendMethod = TransferServiceGrpc.getSendMethod) == null) {
      synchronized (TransferServiceGrpc.class) {
        if ((getSendMethod = TransferServiceGrpc.getSendMethod) == null) {
          TransferServiceGrpc.getSendMethod = getSendMethod =
              io.grpc.MethodDescriptor.<com.webank.eggroll.core.transfer.Transfer.TransferBatch, com.webank.eggroll.core.transfer.Transfer.TransferBatch>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.CLIENT_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "send"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.transfer.Transfer.TransferBatch.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.transfer.Transfer.TransferBatch.getDefaultInstance()))
              .setSchemaDescriptor(new TransferServiceMethodDescriptorSupplier("send"))
              .build();
        }
      }
    }
    return getSendMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch,
      com.webank.eggroll.core.transfer.Transfer.TransferBatch> getRecvMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "recv",
      requestType = com.webank.eggroll.core.transfer.Transfer.TransferBatch.class,
      responseType = com.webank.eggroll.core.transfer.Transfer.TransferBatch.class,
      methodType = io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch,
      com.webank.eggroll.core.transfer.Transfer.TransferBatch> getRecvMethod() {
    io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch, com.webank.eggroll.core.transfer.Transfer.TransferBatch> getRecvMethod;
    if ((getRecvMethod = TransferServiceGrpc.getRecvMethod) == null) {
      synchronized (TransferServiceGrpc.class) {
        if ((getRecvMethod = TransferServiceGrpc.getRecvMethod) == null) {
          TransferServiceGrpc.getRecvMethod = getRecvMethod =
              io.grpc.MethodDescriptor.<com.webank.eggroll.core.transfer.Transfer.TransferBatch, com.webank.eggroll.core.transfer.Transfer.TransferBatch>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "recv"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.transfer.Transfer.TransferBatch.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.transfer.Transfer.TransferBatch.getDefaultInstance()))
              .setSchemaDescriptor(new TransferServiceMethodDescriptorSupplier("recv"))
              .build();
        }
      }
    }
    return getRecvMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch,
      com.webank.eggroll.core.transfer.Transfer.TransferBatch> getSendRecvMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "sendRecv",
      requestType = com.webank.eggroll.core.transfer.Transfer.TransferBatch.class,
      responseType = com.webank.eggroll.core.transfer.Transfer.TransferBatch.class,
      methodType = io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch,
      com.webank.eggroll.core.transfer.Transfer.TransferBatch> getSendRecvMethod() {
    io.grpc.MethodDescriptor<com.webank.eggroll.core.transfer.Transfer.TransferBatch, com.webank.eggroll.core.transfer.Transfer.TransferBatch> getSendRecvMethod;
    if ((getSendRecvMethod = TransferServiceGrpc.getSendRecvMethod) == null) {
      synchronized (TransferServiceGrpc.class) {
        if ((getSendRecvMethod = TransferServiceGrpc.getSendRecvMethod) == null) {
          TransferServiceGrpc.getSendRecvMethod = getSendRecvMethod =
              io.grpc.MethodDescriptor.<com.webank.eggroll.core.transfer.Transfer.TransferBatch, com.webank.eggroll.core.transfer.Transfer.TransferBatch>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "sendRecv"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.transfer.Transfer.TransferBatch.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.transfer.Transfer.TransferBatch.getDefaultInstance()))
              .setSchemaDescriptor(new TransferServiceMethodDescriptorSupplier("sendRecv"))
              .build();
        }
      }
    }
    return getSendRecvMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static TransferServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TransferServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TransferServiceStub>() {
        @java.lang.Override
        public TransferServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TransferServiceStub(channel, callOptions);
        }
      };
    return TransferServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static TransferServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TransferServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TransferServiceBlockingStub>() {
        @java.lang.Override
        public TransferServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TransferServiceBlockingStub(channel, callOptions);
        }
      };
    return TransferServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static TransferServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<TransferServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<TransferServiceFutureStub>() {
        @java.lang.Override
        public TransferServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new TransferServiceFutureStub(channel, callOptions);
        }
      };
    return TransferServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * TODO: use transfer lib
   * </pre>
   */
  public interface AsyncService {

    /**
     */
    default io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> send(
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getSendMethod(), responseObserver);
    }

    /**
     */
    default void recv(com.webank.eggroll.core.transfer.Transfer.TransferBatch request,
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRecvMethod(), responseObserver);
    }

    /**
     */
    default io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> sendRecv(
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getSendRecvMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service TransferService.
   * <pre>
   * TODO: use transfer lib
   * </pre>
   */
  public static abstract class TransferServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return TransferServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service TransferService.
   * <pre>
   * TODO: use transfer lib
   * </pre>
   */
  public static final class TransferServiceStub
      extends io.grpc.stub.AbstractAsyncStub<TransferServiceStub> {
    private TransferServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TransferServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TransferServiceStub(channel, callOptions);
    }

    /**
     */
    public io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> send(
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncClientStreamingCall(
          getChannel().newCall(getSendMethod(), getCallOptions()), responseObserver);
    }

    /**
     */
    public void recv(com.webank.eggroll.core.transfer.Transfer.TransferBatch request,
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> responseObserver) {
      io.grpc.stub.ClientCalls.asyncServerStreamingCall(
          getChannel().newCall(getRecvMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> sendRecv(
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncBidiStreamingCall(
          getChannel().newCall(getSendRecvMethod(), getCallOptions()), responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service TransferService.
   * <pre>
   * TODO: use transfer lib
   * </pre>
   */
  public static final class TransferServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<TransferServiceBlockingStub> {
    private TransferServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TransferServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TransferServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public java.util.Iterator<com.webank.eggroll.core.transfer.Transfer.TransferBatch> recv(
        com.webank.eggroll.core.transfer.Transfer.TransferBatch request) {
      return io.grpc.stub.ClientCalls.blockingServerStreamingCall(
          getChannel(), getRecvMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service TransferService.
   * <pre>
   * TODO: use transfer lib
   * </pre>
   */
  public static final class TransferServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<TransferServiceFutureStub> {
    private TransferServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected TransferServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new TransferServiceFutureStub(channel, callOptions);
    }
  }

  private static final int METHODID_RECV = 0;
  private static final int METHODID_SEND = 1;
  private static final int METHODID_SEND_RECV = 2;

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
        case METHODID_RECV:
          serviceImpl.recv((com.webank.eggroll.core.transfer.Transfer.TransferBatch) request,
              (io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch>) responseObserver);
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
        case METHODID_SEND:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.send(
              (io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch>) responseObserver);
        case METHODID_SEND_RECV:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.sendRecv(
              (io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch>) responseObserver);
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getSendMethod(),
          io.grpc.stub.ServerCalls.asyncClientStreamingCall(
            new MethodHandlers<
              com.webank.eggroll.core.transfer.Transfer.TransferBatch,
              com.webank.eggroll.core.transfer.Transfer.TransferBatch>(
                service, METHODID_SEND)))
        .addMethod(
          getRecvMethod(),
          io.grpc.stub.ServerCalls.asyncServerStreamingCall(
            new MethodHandlers<
              com.webank.eggroll.core.transfer.Transfer.TransferBatch,
              com.webank.eggroll.core.transfer.Transfer.TransferBatch>(
                service, METHODID_RECV)))
        .addMethod(
          getSendRecvMethod(),
          io.grpc.stub.ServerCalls.asyncBidiStreamingCall(
            new MethodHandlers<
              com.webank.eggroll.core.transfer.Transfer.TransferBatch,
              com.webank.eggroll.core.transfer.Transfer.TransferBatch>(
                service, METHODID_SEND_RECV)))
        .build();
  }

  private static abstract class TransferServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    TransferServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.eggroll.core.transfer.Transfer.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("TransferService");
    }
  }

  private static final class TransferServiceFileDescriptorSupplier
      extends TransferServiceBaseDescriptorSupplier {
    TransferServiceFileDescriptorSupplier() {}
  }

  private static final class TransferServiceMethodDescriptorSupplier
      extends TransferServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    TransferServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (TransferServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new TransferServiceFileDescriptorSupplier())
              .addMethod(getSendMethod())
              .addMethod(getRecvMethod())
              .addMethod(getSendRecvMethod())
              .build();
        }
      }
    }
    return result;
  }
}
