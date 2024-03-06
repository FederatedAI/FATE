package com.webank.eggroll.core.command;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/command.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class CommandServiceGrpc {

  private CommandServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.eggroll.core.command.CommandService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.eggroll.core.command.Command.CommandRequest,
      com.webank.eggroll.core.command.Command.CommandResponse> getCallMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "call",
      requestType = com.webank.eggroll.core.command.Command.CommandRequest.class,
      responseType = com.webank.eggroll.core.command.Command.CommandResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.eggroll.core.command.Command.CommandRequest,
      com.webank.eggroll.core.command.Command.CommandResponse> getCallMethod() {
    io.grpc.MethodDescriptor<com.webank.eggroll.core.command.Command.CommandRequest, com.webank.eggroll.core.command.Command.CommandResponse> getCallMethod;
    if ((getCallMethod = CommandServiceGrpc.getCallMethod) == null) {
      synchronized (CommandServiceGrpc.class) {
        if ((getCallMethod = CommandServiceGrpc.getCallMethod) == null) {
          CommandServiceGrpc.getCallMethod = getCallMethod =
              io.grpc.MethodDescriptor.<com.webank.eggroll.core.command.Command.CommandRequest, com.webank.eggroll.core.command.Command.CommandResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "call"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.command.Command.CommandRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.command.Command.CommandResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CommandServiceMethodDescriptorSupplier("call"))
              .build();
        }
      }
    }
    return getCallMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.eggroll.core.command.Command.CommandRequest,
      com.webank.eggroll.core.command.Command.CommandResponse> getCallStreamMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "callStream",
      requestType = com.webank.eggroll.core.command.Command.CommandRequest.class,
      responseType = com.webank.eggroll.core.command.Command.CommandResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.eggroll.core.command.Command.CommandRequest,
      com.webank.eggroll.core.command.Command.CommandResponse> getCallStreamMethod() {
    io.grpc.MethodDescriptor<com.webank.eggroll.core.command.Command.CommandRequest, com.webank.eggroll.core.command.Command.CommandResponse> getCallStreamMethod;
    if ((getCallStreamMethod = CommandServiceGrpc.getCallStreamMethod) == null) {
      synchronized (CommandServiceGrpc.class) {
        if ((getCallStreamMethod = CommandServiceGrpc.getCallStreamMethod) == null) {
          CommandServiceGrpc.getCallStreamMethod = getCallStreamMethod =
              io.grpc.MethodDescriptor.<com.webank.eggroll.core.command.Command.CommandRequest, com.webank.eggroll.core.command.Command.CommandResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "callStream"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.command.Command.CommandRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.eggroll.core.command.Command.CommandResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CommandServiceMethodDescriptorSupplier("callStream"))
              .build();
        }
      }
    }
    return getCallStreamMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static CommandServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CommandServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CommandServiceStub>() {
        @java.lang.Override
        public CommandServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CommandServiceStub(channel, callOptions);
        }
      };
    return CommandServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static CommandServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CommandServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CommandServiceBlockingStub>() {
        @java.lang.Override
        public CommandServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CommandServiceBlockingStub(channel, callOptions);
        }
      };
    return CommandServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static CommandServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CommandServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CommandServiceFutureStub>() {
        @java.lang.Override
        public CommandServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CommandServiceFutureStub(channel, callOptions);
        }
      };
    return CommandServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void call(com.webank.eggroll.core.command.Command.CommandRequest request,
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCallMethod(), responseObserver);
    }

    /**
     */
    default io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandRequest> callStream(
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandResponse> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getCallStreamMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service CommandService.
   */
  public static abstract class CommandServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return CommandServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service CommandService.
   */
  public static final class CommandServiceStub
      extends io.grpc.stub.AbstractAsyncStub<CommandServiceStub> {
    private CommandServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CommandServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CommandServiceStub(channel, callOptions);
    }

    /**
     */
    public void call(com.webank.eggroll.core.command.Command.CommandRequest request,
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCallMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandRequest> callStream(
        io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandResponse> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncBidiStreamingCall(
          getChannel().newCall(getCallStreamMethod(), getCallOptions()), responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service CommandService.
   */
  public static final class CommandServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<CommandServiceBlockingStub> {
    private CommandServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CommandServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CommandServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.webank.eggroll.core.command.Command.CommandResponse call(com.webank.eggroll.core.command.Command.CommandRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCallMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service CommandService.
   */
  public static final class CommandServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<CommandServiceFutureStub> {
    private CommandServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CommandServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CommandServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.eggroll.core.command.Command.CommandResponse> call(
        com.webank.eggroll.core.command.Command.CommandRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCallMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_CALL = 0;
  private static final int METHODID_CALL_STREAM = 1;

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
        case METHODID_CALL:
          serviceImpl.call((com.webank.eggroll.core.command.Command.CommandRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandResponse>) responseObserver);
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
        case METHODID_CALL_STREAM:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.callStream(
              (io.grpc.stub.StreamObserver<com.webank.eggroll.core.command.Command.CommandResponse>) responseObserver);
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getCallMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.eggroll.core.command.Command.CommandRequest,
              com.webank.eggroll.core.command.Command.CommandResponse>(
                service, METHODID_CALL)))
        .addMethod(
          getCallStreamMethod(),
          io.grpc.stub.ServerCalls.asyncBidiStreamingCall(
            new MethodHandlers<
              com.webank.eggroll.core.command.Command.CommandRequest,
              com.webank.eggroll.core.command.Command.CommandResponse>(
                service, METHODID_CALL_STREAM)))
        .build();
  }

  private static abstract class CommandServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    CommandServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.eggroll.core.command.Command.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("CommandService");
    }
  }

  private static final class CommandServiceFileDescriptorSupplier
      extends CommandServiceBaseDescriptorSupplier {
    CommandServiceFileDescriptorSupplier() {}
  }

  private static final class CommandServiceMethodDescriptorSupplier
      extends CommandServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    CommandServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (CommandServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new CommandServiceFileDescriptorSupplier())
              .addMethod(getCallMethod())
              .addMethod(getCallStreamMethod())
              .build();
        }
      }
    }
    return result;
  }
}
