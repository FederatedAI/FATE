package org.ppc.ptp;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: osx.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class PrivateTransferProtocolGrpc {

  private PrivateTransferProtocolGrpc() {}

  public static final java.lang.String SERVICE_NAME = "org.ppc.ptp.PrivateTransferProtocol";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<org.ppc.ptp.Osx.Inbound,
      org.ppc.ptp.Osx.Outbound> getTransportMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "transport",
      requestType = org.ppc.ptp.Osx.Inbound.class,
      responseType = org.ppc.ptp.Osx.Outbound.class,
      methodType = io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
  public static io.grpc.MethodDescriptor<org.ppc.ptp.Osx.Inbound,
      org.ppc.ptp.Osx.Outbound> getTransportMethod() {
    io.grpc.MethodDescriptor<org.ppc.ptp.Osx.Inbound, org.ppc.ptp.Osx.Outbound> getTransportMethod;
    if ((getTransportMethod = PrivateTransferProtocolGrpc.getTransportMethod) == null) {
      synchronized (PrivateTransferProtocolGrpc.class) {
        if ((getTransportMethod = PrivateTransferProtocolGrpc.getTransportMethod) == null) {
          PrivateTransferProtocolGrpc.getTransportMethod = getTransportMethod =
              io.grpc.MethodDescriptor.<org.ppc.ptp.Osx.Inbound, org.ppc.ptp.Osx.Outbound>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "transport"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.Inbound.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.Outbound.getDefaultInstance()))
              .setSchemaDescriptor(new PrivateTransferProtocolMethodDescriptorSupplier("transport"))
              .build();
        }
      }
    }
    return getTransportMethod;
  }

  private static volatile io.grpc.MethodDescriptor<org.ppc.ptp.Osx.Inbound,
      org.ppc.ptp.Osx.Outbound> getInvokeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "invoke",
      requestType = org.ppc.ptp.Osx.Inbound.class,
      responseType = org.ppc.ptp.Osx.Outbound.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<org.ppc.ptp.Osx.Inbound,
      org.ppc.ptp.Osx.Outbound> getInvokeMethod() {
    io.grpc.MethodDescriptor<org.ppc.ptp.Osx.Inbound, org.ppc.ptp.Osx.Outbound> getInvokeMethod;
    if ((getInvokeMethod = PrivateTransferProtocolGrpc.getInvokeMethod) == null) {
      synchronized (PrivateTransferProtocolGrpc.class) {
        if ((getInvokeMethod = PrivateTransferProtocolGrpc.getInvokeMethod) == null) {
          PrivateTransferProtocolGrpc.getInvokeMethod = getInvokeMethod =
              io.grpc.MethodDescriptor.<org.ppc.ptp.Osx.Inbound, org.ppc.ptp.Osx.Outbound>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "invoke"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.Inbound.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  org.ppc.ptp.Osx.Outbound.getDefaultInstance()))
              .setSchemaDescriptor(new PrivateTransferProtocolMethodDescriptorSupplier("invoke"))
              .build();
        }
      }
    }
    return getInvokeMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static PrivateTransferProtocolStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PrivateTransferProtocolStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PrivateTransferProtocolStub>() {
        @java.lang.Override
        public PrivateTransferProtocolStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PrivateTransferProtocolStub(channel, callOptions);
        }
      };
    return PrivateTransferProtocolStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static PrivateTransferProtocolBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PrivateTransferProtocolBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PrivateTransferProtocolBlockingStub>() {
        @java.lang.Override
        public PrivateTransferProtocolBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PrivateTransferProtocolBlockingStub(channel, callOptions);
        }
      };
    return PrivateTransferProtocolBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static PrivateTransferProtocolFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<PrivateTransferProtocolFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<PrivateTransferProtocolFutureStub>() {
        @java.lang.Override
        public PrivateTransferProtocolFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new PrivateTransferProtocolFutureStub(channel, callOptions);
        }
      };
    return PrivateTransferProtocolFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Inbound> transport(
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Outbound> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getTransportMethod(), responseObserver);
    }

    /**
     * <pre>
     *  rpc clusterTokenApply(stream Inbound)  returns (stream Outbound);
     *  rpc clusterTopicApply(stream Inbound)  returns (stream Outbound);
     * </pre>
     */
    default void invoke(org.ppc.ptp.Osx.Inbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Outbound> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getInvokeMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service PrivateTransferProtocol.
   */
  public static abstract class PrivateTransferProtocolImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return PrivateTransferProtocolGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service PrivateTransferProtocol.
   */
  public static final class PrivateTransferProtocolStub
      extends io.grpc.stub.AbstractAsyncStub<PrivateTransferProtocolStub> {
    private PrivateTransferProtocolStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected PrivateTransferProtocolStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PrivateTransferProtocolStub(channel, callOptions);
    }

    /**
     */
    public io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Inbound> transport(
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Outbound> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncBidiStreamingCall(
          getChannel().newCall(getTransportMethod(), getCallOptions()), responseObserver);
    }

    /**
     * <pre>
     *  rpc clusterTokenApply(stream Inbound)  returns (stream Outbound);
     *  rpc clusterTopicApply(stream Inbound)  returns (stream Outbound);
     * </pre>
     */
    public void invoke(org.ppc.ptp.Osx.Inbound request,
        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Outbound> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getInvokeMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service PrivateTransferProtocol.
   */
  public static final class PrivateTransferProtocolBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<PrivateTransferProtocolBlockingStub> {
    private PrivateTransferProtocolBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected PrivateTransferProtocolBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PrivateTransferProtocolBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     *  rpc clusterTokenApply(stream Inbound)  returns (stream Outbound);
     *  rpc clusterTopicApply(stream Inbound)  returns (stream Outbound);
     * </pre>
     */
    public org.ppc.ptp.Osx.Outbound invoke(org.ppc.ptp.Osx.Inbound request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getInvokeMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service PrivateTransferProtocol.
   */
  public static final class PrivateTransferProtocolFutureStub
      extends io.grpc.stub.AbstractFutureStub<PrivateTransferProtocolFutureStub> {
    private PrivateTransferProtocolFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected PrivateTransferProtocolFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new PrivateTransferProtocolFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     *  rpc clusterTokenApply(stream Inbound)  returns (stream Outbound);
     *  rpc clusterTopicApply(stream Inbound)  returns (stream Outbound);
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<org.ppc.ptp.Osx.Outbound> invoke(
        org.ppc.ptp.Osx.Inbound request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getInvokeMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_INVOKE = 0;
  private static final int METHODID_TRANSPORT = 1;

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
        case METHODID_INVOKE:
          serviceImpl.invoke((org.ppc.ptp.Osx.Inbound) request,
              (io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Outbound>) responseObserver);
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
        case METHODID_TRANSPORT:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.transport(
              (io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.Outbound>) responseObserver);
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getTransportMethod(),
          io.grpc.stub.ServerCalls.asyncBidiStreamingCall(
            new MethodHandlers<
              org.ppc.ptp.Osx.Inbound,
              org.ppc.ptp.Osx.Outbound>(
                service, METHODID_TRANSPORT)))
        .addMethod(
          getInvokeMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              org.ppc.ptp.Osx.Inbound,
              org.ppc.ptp.Osx.Outbound>(
                service, METHODID_INVOKE)))
        .build();
  }

  private static abstract class PrivateTransferProtocolBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    PrivateTransferProtocolBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return org.ppc.ptp.Osx.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("PrivateTransferProtocol");
    }
  }

  private static final class PrivateTransferProtocolFileDescriptorSupplier
      extends PrivateTransferProtocolBaseDescriptorSupplier {
    PrivateTransferProtocolFileDescriptorSupplier() {}
  }

  private static final class PrivateTransferProtocolMethodDescriptorSupplier
      extends PrivateTransferProtocolBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    PrivateTransferProtocolMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (PrivateTransferProtocolGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new PrivateTransferProtocolFileDescriptorSupplier())
              .addMethod(getTransportMethod())
              .addMethod(getInvokeMethod())
              .build();
        }
      }
    }
    return result;
  }
}
