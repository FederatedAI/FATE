package org.ppc.ptp;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: osx.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class PrivateTransferTransportGrpc {

  private PrivateTransferTransportGrpc() {}

  public static final java.lang.String SERVICE_NAME = "org.ppc.ptp.PrivateTransferTransport";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PeekInbound,
      org.ppc.ptp.Osx.TransportOutbound> getPeekMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "peek",
      requestType = org.ppc.ptp.Osx.PeekInbound.class,
      responseType = org.ppc.ptp.Osx.TransportOutbound.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PeekInbound,
      org.ppc.ptp.Osx.TransportOutbound> getPeekMethod() {
    io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PeekInbound, org.ppc.ptp.Osx.TransportOutbound> getPeekMethod;
    if ((getPeekMethod = PrivateTransferTransportGrpc.getPeekMethod) == null) {
      synchronized (PrivateTransferTransportGrpc.class) {
        if ((getPeekMethod = PrivateTransferTransportGrpc.getPeekMethod) == null) {
          PrivateTransferTransportGrpc.getPeekMethod = getPeekMethod =
              io.grpc.MethodDescriptor.<org.ppc.ptp.Osx.PeekInbound, org.ppc.ptp.Osx.TransportOutbound>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "peek"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.PeekInbound.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.TransportOutbound.getDefaultInstance()))
              .setSchemaDescriptor(new PrivateTransferTransportMethodDescriptorSupplier("peek"))
              .build();
        }
      }
    }
    return getPeekMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PopInbound,
      org.ppc.ptp.Osx.TransportOutbound> getPopMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "pop",
      requestType = org.ppc.ptp.Osx.PopInbound.class,
      responseType = org.ppc.ptp.Osx.TransportOutbound.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PopInbound,
      org.ppc.ptp.Osx.TransportOutbound> getPopMethod() {
    io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PopInbound, org.ppc.ptp.Osx.TransportOutbound> getPopMethod;
    if ((getPopMethod = PrivateTransferTransportGrpc.getPopMethod) == null) {
      synchronized (PrivateTransferTransportGrpc.class) {
        if ((getPopMethod = PrivateTransferTransportGrpc.getPopMethod) == null) {
          PrivateTransferTransportGrpc.getPopMethod = getPopMethod =
              io.grpc.MethodDescriptor.<org.ppc.ptp.Osx.PopInbound, org.ppc.ptp.Osx.TransportOutbound>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "pop"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.PopInbound.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.TransportOutbound.getDefaultInstance()))
              .setSchemaDescriptor(new PrivateTransferTransportMethodDescriptorSupplier("pop"))
              .build();
        }
      }
    }
    return getPopMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PushInbound,
      org.ppc.ptp.Osx.TransportOutbound> getPushMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "push",
      requestType = org.ppc.ptp.Osx.PushInbound.class,
      responseType = org.ppc.ptp.Osx.TransportOutbound.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PushInbound,
      org.ppc.ptp.Osx.TransportOutbound> getPushMethod() {
    io.grpc.MethodDescriptor<org.ppc.ptp.Osx.PushInbound, org.ppc.ptp.Osx.TransportOutbound> getPushMethod;
    if ((getPushMethod = PrivateTransferTransportGrpc.getPushMethod) == null) {
      synchronized (PrivateTransferTransportGrpc.class) {
        if ((getPushMethod = PrivateTransferTransportGrpc.getPushMethod) == null) {
          PrivateTransferTransportGrpc.getPushMethod = getPushMethod =
              io.grpc.MethodDescriptor.<org.ppc.ptp.Osx.PushInbound, org.ppc.ptp.Osx.TransportOutbound>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "push"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.PushInbound.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.TransportOutbound.getDefaultInstance()))
              .setSchemaDescriptor(new PrivateTransferTransportMethodDescriptorSupplier("push"))
              .build();
        }
      }
    }
    return getPushMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.ppc.ptp.Osx.ReleaseInbound,
      org.ppc.ptp.Osx.TransportOutbound> getReleaseMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "release",
      requestType = org.ppc.ptp.Osx.ReleaseInbound.class,
      responseType = org.ppc.ptp.Osx.TransportOutbound.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.ppc.ptp.Osx.ReleaseInbound,
      org.ppc.ptp.Osx.TransportOutbound> getReleaseMethod() {
    io.grpc.MethodDescriptor<org.ppc.ptp.Osx.ReleaseInbound, org.ppc.ptp.Osx.TransportOutbound> getReleaseMethod;
    if ((getReleaseMethod = PrivateTransferTransportGrpc.getReleaseMethod) == null) {
      synchronized (PrivateTransferTransportGrpc.class) {
        if ((getReleaseMethod = PrivateTransferTransportGrpc.getReleaseMethod) == null) {
          PrivateTransferTransportGrpc.getReleaseMethod = getReleaseMethod =
              io.grpc.MethodDescriptor.<org.ppc.ptp.Osx.ReleaseInbound, org.ppc.ptp.Osx.TransportOutbound>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "release"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.ReleaseInbound.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.TransportOutbound.getDefaultInstance()))
              .setSchemaDescriptor(new PrivateTransferTransportMethodDescriptorSupplier("release"))
              .build();
        }
      }
    }
    return getReleaseMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static PrivateTransferTransportStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PrivateTransferTransportStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PrivateTransferTransportStub>() {
        @java.lang.Override
        public PrivateTransferTransportStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PrivateTransferTransportStub(channel, callOptions);
        }
      };
    return PrivateTransferTransportStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static PrivateTransferTransportBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PrivateTransferTransportBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PrivateTransferTransportBlockingStub>() {
        @java.lang.Override
        public PrivateTransferTransportBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PrivateTransferTransportBlockingStub(channel, callOptions);
        }
      };
    return PrivateTransferTransportBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static PrivateTransferTransportFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PrivateTransferTransportFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PrivateTransferTransportFutureStub>() {
        @java.lang.Override
        public PrivateTransferTransportFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PrivateTransferTransportFutureStub(channel, callOptions);
        }
      };
    return PrivateTransferTransportFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void peek(org.ppc.ptp.Osx.PeekInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPeekMethod(), responseObserver);
    }

    /**
     */
    default void pop(org.ppc.ptp.Osx.PopInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPopMethod(), responseObserver);
    }

    /**
     */
    default void push(org.ppc.ptp.Osx.PushInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPushMethod(), responseObserver);
    }

    /**
     */
    default void release(org.ppc.ptp.Osx.ReleaseInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getReleaseMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service PrivateTransferTransport.
   */
  public static abstract class PrivateTransferTransportImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return PrivateTransferTransportGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service PrivateTransferTransport.
   */
  public static final class PrivateTransferTransportStub
      extends io.grpc.stub.AbstractAsyncStub<PrivateTransferTransportStub> {
    private PrivateTransferTransportStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected PrivateTransferTransportStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PrivateTransferTransportStub(channel, callOptions);
    }

    /**
     */
    public void peek(org.ppc.ptp.Osx.PeekInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPeekMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void pop(org.ppc.ptp.Osx.PopInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPopMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void push(org.ppc.ptp.Osx.PushInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPushMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void release(org.ppc.ptp.Osx.ReleaseInbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getReleaseMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service PrivateTransferTransport.
   */
  public static final class PrivateTransferTransportBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<PrivateTransferTransportBlockingStub> {
    private PrivateTransferTransportBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected PrivateTransferTransportBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PrivateTransferTransportBlockingStub(channel, callOptions);
    }

    /**
     */
    public org.ppc.ptp.Osx.TransportOutbound peek(org.ppc.ptp.Osx.PeekInbound request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPeekMethod(), getCallOptions(), request);
    }

    /**
     */
    public org.ppc.ptp.Osx.TransportOutbound pop(org.ppc.ptp.Osx.PopInbound request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPopMethod(), getCallOptions(), request);
    }

    /**
     */
    public org.ppc.ptp.Osx.TransportOutbound push(org.ppc.ptp.Osx.PushInbound request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPushMethod(), getCallOptions(), request);
    }

    /**
     */
    public org.ppc.ptp.Osx.TransportOutbound release(org.ppc.ptp.Osx.ReleaseInbound request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getReleaseMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service PrivateTransferTransport.
   */
  public static final class PrivateTransferTransportFutureStub
      extends io.grpc.stub.AbstractFutureStub<PrivateTransferTransportFutureStub> {
    private PrivateTransferTransportFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected PrivateTransferTransportFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PrivateTransferTransportFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<org.ppc.ptp.Osx.TransportOutbound> peek(
        org.ppc.ptp.Osx.PeekInbound request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPeekMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<org.ppc.ptp.Osx.TransportOutbound> pop(
        org.ppc.ptp.Osx.PopInbound request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPopMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<org.ppc.ptp.Osx.TransportOutbound> push(
        org.ppc.ptp.Osx.PushInbound request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPushMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<org.ppc.ptp.Osx.TransportOutbound> release(
        org.ppc.ptp.Osx.ReleaseInbound request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getReleaseMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_PEEK = 0;
  private static final int METHODID_POP = 1;
  private static final int METHODID_PUSH = 2;
  private static final int METHODID_RELEASE = 3;

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
        case METHODID_PEEK:
          serviceImpl.peek((org.ppc.ptp.Osx.PeekInbound) request,
              (io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound>) responseObserver);
          break;
        case METHODID_POP:
          serviceImpl.pop((org.ppc.ptp.Osx.PopInbound) request,
              (io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound>) responseObserver);
          break;
        case METHODID_PUSH:
          serviceImpl.push((org.ppc.ptp.Osx.PushInbound) request,
              (io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound>) responseObserver);
          break;
        case METHODID_RELEASE:
          serviceImpl.release((org.ppc.ptp.Osx.ReleaseInbound) request,
              (io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound>) responseObserver);
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
          getPeekMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              org.ppc.ptp.Osx.PeekInbound,
              org.ppc.ptp.Osx.TransportOutbound>(
                service, METHODID_PEEK)))
        .addMethod(
          getPopMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              org.ppc.ptp.Osx.PopInbound,
              org.ppc.ptp.Osx.TransportOutbound>(
                service, METHODID_POP)))
        .addMethod(
          getPushMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              org.ppc.ptp.Osx.PushInbound,
              org.ppc.ptp.Osx.TransportOutbound>(
                service, METHODID_PUSH)))
        .addMethod(
          getReleaseMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              org.ppc.ptp.Osx.ReleaseInbound,
              org.ppc.ptp.Osx.TransportOutbound>(
                service, METHODID_RELEASE)))
        .build();
  }

  private static abstract class PrivateTransferTransportBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    PrivateTransferTransportBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return org.ppc.ptp.Osx.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("PrivateTransferTransport");
    }
  }

  private static final class PrivateTransferTransportFileDescriptorSupplier
      extends PrivateTransferTransportBaseDescriptorSupplier {
    PrivateTransferTransportFileDescriptorSupplier() {}
  }

  private static final class PrivateTransferTransportMethodDescriptorSupplier
      extends PrivateTransferTransportBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    PrivateTransferTransportMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (PrivateTransferTransportGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new PrivateTransferTransportFileDescriptorSupplier())
              .addMethod(getPeekMethod())
              .addMethod(getPopMethod())
              .addMethod(getPushMethod())
              .addMethod(getReleaseMethod())
              .build();
        }
      }
    }
    return result;
  }
}
