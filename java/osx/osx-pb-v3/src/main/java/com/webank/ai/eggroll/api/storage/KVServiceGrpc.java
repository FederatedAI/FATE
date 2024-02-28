package com.webank.ai.eggroll.api.storage;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * service for actual storage operation
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/kv.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class KVServiceGrpc {

  private KVServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.eggroll.api.storage.KVService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo,
      com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo> getCreateIfAbsentMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "createIfAbsent",
      requestType = com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo,
      com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo> getCreateIfAbsentMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo, com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo> getCreateIfAbsentMethod;
    if ((getCreateIfAbsentMethod = KVServiceGrpc.getCreateIfAbsentMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getCreateIfAbsentMethod = KVServiceGrpc.getCreateIfAbsentMethod) == null) {
          KVServiceGrpc.getCreateIfAbsentMethod = getCreateIfAbsentMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo, com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "createIfAbsent"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("createIfAbsent"))
              .build();
        }
      }
    }
    return getCreateIfAbsentMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getPutMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "put",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getPutMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Empty> getPutMethod;
    if ((getPutMethod = KVServiceGrpc.getPutMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getPutMethod = KVServiceGrpc.getPutMethod) == null) {
          KVServiceGrpc.getPutMethod = getPutMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "put"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("put"))
              .build();
        }
      }
    }
    return getPutMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getPutIfAbsentMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "putIfAbsent",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getPutIfAbsentMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Operand> getPutIfAbsentMethod;
    if ((getPutIfAbsentMethod = KVServiceGrpc.getPutIfAbsentMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getPutIfAbsentMethod = KVServiceGrpc.getPutIfAbsentMethod) == null) {
          KVServiceGrpc.getPutIfAbsentMethod = getPutIfAbsentMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Operand>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "putIfAbsent"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("putIfAbsent"))
              .build();
        }
      }
    }
    return getPutIfAbsentMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getPutAllMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "putAll",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.CLIENT_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getPutAllMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Empty> getPutAllMethod;
    if ((getPutAllMethod = KVServiceGrpc.getPutAllMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getPutAllMethod = KVServiceGrpc.getPutAllMethod) == null) {
          KVServiceGrpc.getPutAllMethod = getPutAllMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.CLIENT_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "putAll"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("putAll"))
              .build();
        }
      }
    }
    return getPutAllMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getDelOneMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "delOne",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getDelOneMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Operand> getDelOneMethod;
    if ((getDelOneMethod = KVServiceGrpc.getDelOneMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getDelOneMethod = KVServiceGrpc.getDelOneMethod) == null) {
          KVServiceGrpc.getDelOneMethod = getDelOneMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Operand>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "delOne"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("delOne"))
              .build();
        }
      }
    }
    return getDelOneMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getGetMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "get",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getGetMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Operand> getGetMethod;
    if ((getGetMethod = KVServiceGrpc.getGetMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getGetMethod = KVServiceGrpc.getGetMethod) == null) {
          KVServiceGrpc.getGetMethod = getGetMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Operand, com.webank.ai.eggroll.api.storage.Kv.Operand>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "get"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("get"))
              .build();
        }
      }
    }
    return getGetMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Range,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getIterateMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "iterate",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Range.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      methodType = io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Range,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getIterateMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Range, com.webank.ai.eggroll.api.storage.Kv.Operand> getIterateMethod;
    if ((getIterateMethod = KVServiceGrpc.getIterateMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getIterateMethod = KVServiceGrpc.getIterateMethod) == null) {
          KVServiceGrpc.getIterateMethod = getIterateMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Range, com.webank.ai.eggroll.api.storage.Kv.Operand>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "iterate"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Range.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("iterate"))
              .build();
        }
      }
    }
    return getIterateMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getDestroyMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "destroy",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Empty.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getDestroyMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty, com.webank.ai.eggroll.api.storage.Kv.Empty> getDestroyMethod;
    if ((getDestroyMethod = KVServiceGrpc.getDestroyMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getDestroyMethod = KVServiceGrpc.getDestroyMethod) == null) {
          KVServiceGrpc.getDestroyMethod = getDestroyMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Empty, com.webank.ai.eggroll.api.storage.Kv.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "destroy"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("destroy"))
              .build();
        }
      }
    }
    return getDestroyMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getDestroyAllMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "destroyAll",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Empty.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Empty.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty,
      com.webank.ai.eggroll.api.storage.Kv.Empty> getDestroyAllMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty, com.webank.ai.eggroll.api.storage.Kv.Empty> getDestroyAllMethod;
    if ((getDestroyAllMethod = KVServiceGrpc.getDestroyAllMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getDestroyAllMethod = KVServiceGrpc.getDestroyAllMethod) == null) {
          KVServiceGrpc.getDestroyAllMethod = getDestroyAllMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Empty, com.webank.ai.eggroll.api.storage.Kv.Empty>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "destroyAll"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Empty.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("destroyAll"))
              .build();
        }
      }
    }
    return getDestroyAllMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty,
      com.webank.ai.eggroll.api.storage.Kv.Count> getCountMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "count",
      requestType = com.webank.ai.eggroll.api.storage.Kv.Empty.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Count.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty,
      com.webank.ai.eggroll.api.storage.Kv.Count> getCountMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.storage.Kv.Empty, com.webank.ai.eggroll.api.storage.Kv.Count> getCountMethod;
    if ((getCountMethod = KVServiceGrpc.getCountMethod) == null) {
      synchronized (KVServiceGrpc.class) {
        if ((getCountMethod = KVServiceGrpc.getCountMethod) == null) {
          KVServiceGrpc.getCountMethod = getCountMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.storage.Kv.Empty, com.webank.ai.eggroll.api.storage.Kv.Count>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "count"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Empty.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Count.getDefaultInstance()))
              .setSchemaDescriptor(new KVServiceMethodDescriptorSupplier("count"))
              .build();
        }
      }
    }
    return getCountMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static KVServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<KVServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<KVServiceStub>() {
        @java.lang.Override
        public KVServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new KVServiceStub(channel, callOptions);
        }
      };
    return KVServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static KVServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<KVServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<KVServiceBlockingStub>() {
        @java.lang.Override
        public KVServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new KVServiceBlockingStub(channel, callOptions);
        }
      };
    return KVServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static KVServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<KVServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<KVServiceFutureStub>() {
        @java.lang.Override
        public KVServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new KVServiceFutureStub(channel, callOptions);
        }
      };
    return KVServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * service for actual storage operation
   * </pre>
   */
  public interface AsyncService {

    /**
     * <pre>
     * create a table
     * </pre>
     */
    default void createIfAbsent(com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCreateIfAbsentMethod(), responseObserver);
    }

    /**
     * <pre>
     * put an entry to table
     * </pre>
     */
    default void put(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPutMethod(), responseObserver);
    }

    /**
     * <pre>
     * put an entry to table if absent
     * </pre>
     */
    default void putIfAbsent(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getPutIfAbsentMethod(), responseObserver);
    }

    /**
     * <pre>
     * put entries to table (entries will be streaming in)
     * </pre>
     */
    default io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> putAll(
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getPutAllMethod(), responseObserver);
    }

    /**
     * <pre>
     * delete an entry from table
     * </pre>
     */
    default void delOne(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDelOneMethod(), responseObserver);
    }

    /**
     * <pre>
     * get an entry from table
     * </pre>
     */
    default void get(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetMethod(), responseObserver);
    }

    /**
     * <pre>
     * iterate through a table. Response entries are ordered
     * </pre>
     */
    default void iterate(com.webank.ai.eggroll.api.storage.Kv.Range request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getIterateMethod(), responseObserver);
    }

    /**
     * <pre>
     * destroy a table
     * </pre>
     */
    default void destroy(com.webank.ai.eggroll.api.storage.Kv.Empty request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDestroyMethod(), responseObserver);
    }

    /**
     * <pre>
     * destroy multiple tables
     * </pre>
     */
    default void destroyAll(com.webank.ai.eggroll.api.storage.Kv.Empty request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getDestroyAllMethod(), responseObserver);
    }

    /**
     * <pre>
     * count record amount of a table
     * </pre>
     */
    default void count(com.webank.ai.eggroll.api.storage.Kv.Empty request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Count> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCountMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service KVService.
   * <pre>
   * service for actual storage operation
   * </pre>
   */
  public static abstract class KVServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return KVServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service KVService.
   * <pre>
   * service for actual storage operation
   * </pre>
   */
  public static final class KVServiceStub
      extends io.grpc.stub.AbstractAsyncStub<KVServiceStub> {
    private KVServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected KVServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new KVServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     * create a table
     * </pre>
     */
    public void createIfAbsent(com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCreateIfAbsentMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * put an entry to table
     * </pre>
     */
    public void put(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPutMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * put an entry to table if absent
     * </pre>
     */
    public void putIfAbsent(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getPutIfAbsentMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * put entries to table (entries will be streaming in)
     * </pre>
     */
    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> putAll(
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncClientStreamingCall(
          getChannel().newCall(getPutAllMethod(), getCallOptions()), responseObserver);
    }

    /**
     * <pre>
     * delete an entry from table
     * </pre>
     */
    public void delOne(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDelOneMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * get an entry from table
     * </pre>
     */
    public void get(com.webank.ai.eggroll.api.storage.Kv.Operand request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * iterate through a table. Response entries are ordered
     * </pre>
     */
    public void iterate(com.webank.ai.eggroll.api.storage.Kv.Range request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ClientCalls.asyncServerStreamingCall(
          getChannel().newCall(getIterateMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * destroy a table
     * </pre>
     */
    public void destroy(com.webank.ai.eggroll.api.storage.Kv.Empty request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDestroyMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * destroy multiple tables
     * </pre>
     */
    public void destroyAll(com.webank.ai.eggroll.api.storage.Kv.Empty request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getDestroyAllMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     * count record amount of a table
     * </pre>
     */
    public void count(com.webank.ai.eggroll.api.storage.Kv.Empty request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Count> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCountMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service KVService.
   * <pre>
   * service for actual storage operation
   * </pre>
   */
  public static final class KVServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<KVServiceBlockingStub> {
    private KVServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected KVServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new KVServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     * create a table
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo createIfAbsent(com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCreateIfAbsentMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * put an entry to table
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.Empty put(com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPutMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * put an entry to table if absent
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.Operand putIfAbsent(com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getPutIfAbsentMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * delete an entry from table
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.Operand delOne(com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDelOneMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * get an entry from table
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.Operand get(com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * iterate through a table. Response entries are ordered
     * </pre>
     */
    public java.util.Iterator<com.webank.ai.eggroll.api.storage.Kv.Operand> iterate(
        com.webank.ai.eggroll.api.storage.Kv.Range request) {
      return io.grpc.stub.ClientCalls.blockingServerStreamingCall(
          getChannel(), getIterateMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * destroy a table
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.Empty destroy(com.webank.ai.eggroll.api.storage.Kv.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDestroyMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * destroy multiple tables
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.Empty destroyAll(com.webank.ai.eggroll.api.storage.Kv.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getDestroyAllMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     * count record amount of a table
     * </pre>
     */
    public com.webank.ai.eggroll.api.storage.Kv.Count count(com.webank.ai.eggroll.api.storage.Kv.Empty request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCountMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service KVService.
   * <pre>
   * service for actual storage operation
   * </pre>
   */
  public static final class KVServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<KVServiceFutureStub> {
    private KVServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected KVServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new KVServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     * create a table
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo> createIfAbsent(
        com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCreateIfAbsentMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * put an entry to table
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.Empty> put(
        com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPutMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * put an entry to table if absent
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.Operand> putIfAbsent(
        com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getPutIfAbsentMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * delete an entry from table
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.Operand> delOne(
        com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDelOneMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * get an entry from table
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.Operand> get(
        com.webank.ai.eggroll.api.storage.Kv.Operand request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * destroy a table
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.Empty> destroy(
        com.webank.ai.eggroll.api.storage.Kv.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDestroyMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * destroy multiple tables
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.Empty> destroyAll(
        com.webank.ai.eggroll.api.storage.Kv.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getDestroyAllMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     * count record amount of a table
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.Kv.Count> count(
        com.webank.ai.eggroll.api.storage.Kv.Empty request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCountMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_CREATE_IF_ABSENT = 0;
  private static final int METHODID_PUT = 1;
  private static final int METHODID_PUT_IF_ABSENT = 2;
  private static final int METHODID_DEL_ONE = 3;
  private static final int METHODID_GET = 4;
  private static final int METHODID_ITERATE = 5;
  private static final int METHODID_DESTROY = 6;
  private static final int METHODID_DESTROY_ALL = 7;
  private static final int METHODID_COUNT = 8;
  private static final int METHODID_PUT_ALL = 9;

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
        case METHODID_CREATE_IF_ABSENT:
          serviceImpl.createIfAbsent((com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo>) responseObserver);
          break;
        case METHODID_PUT:
          serviceImpl.put((com.webank.ai.eggroll.api.storage.Kv.Operand) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty>) responseObserver);
          break;
        case METHODID_PUT_IF_ABSENT:
          serviceImpl.putIfAbsent((com.webank.ai.eggroll.api.storage.Kv.Operand) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand>) responseObserver);
          break;
        case METHODID_DEL_ONE:
          serviceImpl.delOne((com.webank.ai.eggroll.api.storage.Kv.Operand) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand>) responseObserver);
          break;
        case METHODID_GET:
          serviceImpl.get((com.webank.ai.eggroll.api.storage.Kv.Operand) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand>) responseObserver);
          break;
        case METHODID_ITERATE:
          serviceImpl.iterate((com.webank.ai.eggroll.api.storage.Kv.Range) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand>) responseObserver);
          break;
        case METHODID_DESTROY:
          serviceImpl.destroy((com.webank.ai.eggroll.api.storage.Kv.Empty) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty>) responseObserver);
          break;
        case METHODID_DESTROY_ALL:
          serviceImpl.destroyAll((com.webank.ai.eggroll.api.storage.Kv.Empty) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty>) responseObserver);
          break;
        case METHODID_COUNT:
          serviceImpl.count((com.webank.ai.eggroll.api.storage.Kv.Empty) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Count>) responseObserver);
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
        case METHODID_PUT_ALL:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.putAll(
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Empty>) responseObserver);
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getCreateIfAbsentMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo,
              com.webank.ai.eggroll.api.storage.Kv.CreateTableInfo>(
                service, METHODID_CREATE_IF_ABSENT)))
        .addMethod(
          getPutMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Operand,
              com.webank.ai.eggroll.api.storage.Kv.Empty>(
                service, METHODID_PUT)))
        .addMethod(
          getPutIfAbsentMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Operand,
              com.webank.ai.eggroll.api.storage.Kv.Operand>(
                service, METHODID_PUT_IF_ABSENT)))
        .addMethod(
          getPutAllMethod(),
          io.grpc.stub.ServerCalls.asyncClientStreamingCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Operand,
              com.webank.ai.eggroll.api.storage.Kv.Empty>(
                service, METHODID_PUT_ALL)))
        .addMethod(
          getDelOneMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Operand,
              com.webank.ai.eggroll.api.storage.Kv.Operand>(
                service, METHODID_DEL_ONE)))
        .addMethod(
          getGetMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Operand,
              com.webank.ai.eggroll.api.storage.Kv.Operand>(
                service, METHODID_GET)))
        .addMethod(
          getIterateMethod(),
          io.grpc.stub.ServerCalls.asyncServerStreamingCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Range,
              com.webank.ai.eggroll.api.storage.Kv.Operand>(
                service, METHODID_ITERATE)))
        .addMethod(
          getDestroyMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Empty,
              com.webank.ai.eggroll.api.storage.Kv.Empty>(
                service, METHODID_DESTROY)))
        .addMethod(
          getDestroyAllMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Empty,
              com.webank.ai.eggroll.api.storage.Kv.Empty>(
                service, METHODID_DESTROY_ALL)))
        .addMethod(
          getCountMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.storage.Kv.Empty,
              com.webank.ai.eggroll.api.storage.Kv.Count>(
                service, METHODID_COUNT)))
        .build();
  }

  private static abstract class KVServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    KVServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.eggroll.api.storage.Kv.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("KVService");
    }
  }

  private static final class KVServiceFileDescriptorSupplier
      extends KVServiceBaseDescriptorSupplier {
    KVServiceFileDescriptorSupplier() {}
  }

  private static final class KVServiceMethodDescriptorSupplier
      extends KVServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    KVServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (KVServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new KVServiceFileDescriptorSupplier())
              .addMethod(getCreateIfAbsentMethod())
              .addMethod(getPutMethod())
              .addMethod(getPutIfAbsentMethod())
              .addMethod(getPutAllMethod())
              .addMethod(getDelOneMethod())
              .addMethod(getGetMethod())
              .addMethod(getIterateMethod())
              .addMethod(getDestroyMethod())
              .addMethod(getDestroyAllMethod())
              .addMethod(getCountMethod())
              .build();
        }
      }
    }
    return result;
  }
}
