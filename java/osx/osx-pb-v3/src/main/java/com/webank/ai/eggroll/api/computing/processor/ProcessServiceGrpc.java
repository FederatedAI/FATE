package com.webank.ai.eggroll.api.computing.processor;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/processor.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class ProcessServiceGrpc {

  private ProcessServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.eggroll.api.computing.processor.ProcessService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "map",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapMethod;
    if ((getMapMethod = ProcessServiceGrpc.getMapMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getMapMethod = ProcessServiceGrpc.getMapMethod) == null) {
          ProcessServiceGrpc.getMapMethod = getMapMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "map"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("map"))
              .build();
        }
      }
    }
    return getMapMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapValuesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "mapValues",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapValuesMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapValuesMethod;
    if ((getMapValuesMethod = ProcessServiceGrpc.getMapValuesMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getMapValuesMethod = ProcessServiceGrpc.getMapValuesMethod) == null) {
          ProcessServiceGrpc.getMapValuesMethod = getMapValuesMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "mapValues"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("mapValues"))
              .build();
        }
      }
    }
    return getMapValuesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getJoinMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "join",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getJoinMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getJoinMethod;
    if ((getJoinMethod = ProcessServiceGrpc.getJoinMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getJoinMethod = ProcessServiceGrpc.getJoinMethod) == null) {
          ProcessServiceGrpc.getJoinMethod = getJoinMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "join"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("join"))
              .build();
        }
      }
    }
    return getJoinMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getReduceMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "reduce",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.Kv.Operand.class,
      methodType = io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.Kv.Operand> getReduceMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.Kv.Operand> getReduceMethod;
    if ((getReduceMethod = ProcessServiceGrpc.getReduceMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getReduceMethod = ProcessServiceGrpc.getReduceMethod) == null) {
          ProcessServiceGrpc.getReduceMethod = getReduceMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.Kv.Operand>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.SERVER_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "reduce"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.Kv.Operand.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("reduce"))
              .build();
        }
      }
    }
    return getReduceMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapPartitionsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "mapPartitions",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapPartitionsMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getMapPartitionsMethod;
    if ((getMapPartitionsMethod = ProcessServiceGrpc.getMapPartitionsMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getMapPartitionsMethod = ProcessServiceGrpc.getMapPartitionsMethod) == null) {
          ProcessServiceGrpc.getMapPartitionsMethod = getMapPartitionsMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "mapPartitions"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("mapPartitions"))
              .build();
        }
      }
    }
    return getMapPartitionsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getGlomMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "glom",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getGlomMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getGlomMethod;
    if ((getGlomMethod = ProcessServiceGrpc.getGlomMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getGlomMethod = ProcessServiceGrpc.getGlomMethod) == null) {
          ProcessServiceGrpc.getGlomMethod = getGlomMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "glom"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("glom"))
              .build();
        }
      }
    }
    return getGlomMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getSampleMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "sample",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getSampleMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getSampleMethod;
    if ((getSampleMethod = ProcessServiceGrpc.getSampleMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getSampleMethod = ProcessServiceGrpc.getSampleMethod) == null) {
          ProcessServiceGrpc.getSampleMethod = getSampleMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "sample"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("sample"))
              .build();
        }
      }
    }
    return getSampleMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getSubtractByKeyMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "subtractByKey",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getSubtractByKeyMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getSubtractByKeyMethod;
    if ((getSubtractByKeyMethod = ProcessServiceGrpc.getSubtractByKeyMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getSubtractByKeyMethod = ProcessServiceGrpc.getSubtractByKeyMethod) == null) {
          ProcessServiceGrpc.getSubtractByKeyMethod = getSubtractByKeyMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "subtractByKey"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("subtractByKey"))
              .build();
        }
      }
    }
    return getSubtractByKeyMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getFilterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "filter",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getFilterMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getFilterMethod;
    if ((getFilterMethod = ProcessServiceGrpc.getFilterMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getFilterMethod = ProcessServiceGrpc.getFilterMethod) == null) {
          ProcessServiceGrpc.getFilterMethod = getFilterMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "filter"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("filter"))
              .build();
        }
      }
    }
    return getFilterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getUnionMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "union",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getUnionMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getUnionMethod;
    if ((getUnionMethod = ProcessServiceGrpc.getUnionMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getUnionMethod = ProcessServiceGrpc.getUnionMethod) == null) {
          ProcessServiceGrpc.getUnionMethod = getUnionMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "union"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("union"))
              .build();
        }
      }
    }
    return getUnionMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getFlatMapMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "flatMap",
      requestType = com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.class,
      responseType = com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
      com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getFlatMapMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> getFlatMapMethod;
    if ((getFlatMapMethod = ProcessServiceGrpc.getFlatMapMethod) == null) {
      synchronized (ProcessServiceGrpc.class) {
        if ((getFlatMapMethod = ProcessServiceGrpc.getFlatMapMethod) == null) {
          ProcessServiceGrpc.getFlatMapMethod = getFlatMapMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess, com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "flatMap"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator.getDefaultInstance()))
              .setSchemaDescriptor(new ProcessServiceMethodDescriptorSupplier("flatMap"))
              .build();
        }
      }
    }
    return getFlatMapMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static ProcessServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ProcessServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ProcessServiceStub>() {
        @java.lang.Override
        public ProcessServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ProcessServiceStub(channel, callOptions);
        }
      };
    return ProcessServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static ProcessServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ProcessServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ProcessServiceBlockingStub>() {
        @java.lang.Override
        public ProcessServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ProcessServiceBlockingStub(channel, callOptions);
        }
      };
    return ProcessServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static ProcessServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<ProcessServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<ProcessServiceFutureStub>() {
        @java.lang.Override
        public ProcessServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new ProcessServiceFutureStub(channel, callOptions);
        }
      };
    return ProcessServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void map(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getMapMethod(), responseObserver);
    }

    /**
     */
    default void mapValues(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getMapValuesMethod(), responseObserver);
    }

    /**
     */
    default void join(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getJoinMethod(), responseObserver);
    }

    /**
     */
    default void reduce(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getReduceMethod(), responseObserver);
    }

    /**
     */
    default void mapPartitions(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getMapPartitionsMethod(), responseObserver);
    }

    /**
     */
    default void glom(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGlomMethod(), responseObserver);
    }

    /**
     */
    default void sample(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSampleMethod(), responseObserver);
    }

    /**
     */
    default void subtractByKey(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSubtractByKeyMethod(), responseObserver);
    }

    /**
     */
    default void filter(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getFilterMethod(), responseObserver);
    }

    /**
     */
    default void union(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUnionMethod(), responseObserver);
    }

    /**
     */
    default void flatMap(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getFlatMapMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service ProcessService.
   */
  public static abstract class ProcessServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return ProcessServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service ProcessService.
   */
  public static final class ProcessServiceStub
      extends io.grpc.stub.AbstractAsyncStub<ProcessServiceStub> {
    private ProcessServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ProcessServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ProcessServiceStub(channel, callOptions);
    }

    /**
     */
    public void map(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getMapMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void mapValues(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getMapValuesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void join(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getJoinMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void reduce(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand> responseObserver) {
      io.grpc.stub.ClientCalls.asyncServerStreamingCall(
          getChannel().newCall(getReduceMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void mapPartitions(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getMapPartitionsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void glom(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGlomMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void sample(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSampleMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void subtractByKey(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSubtractByKeyMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void filter(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getFilterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void union(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUnionMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void flatMap(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getFlatMapMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service ProcessService.
   */
  public static final class ProcessServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<ProcessServiceBlockingStub> {
    private ProcessServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ProcessServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ProcessServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator map(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getMapMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator mapValues(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getMapValuesMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator join(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getJoinMethod(), getCallOptions(), request);
    }

    /**
     */
    public java.util.Iterator<com.webank.ai.eggroll.api.storage.Kv.Operand> reduce(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingServerStreamingCall(
          getChannel(), getReduceMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator mapPartitions(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getMapPartitionsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator glom(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGlomMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator sample(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSampleMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator subtractByKey(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSubtractByKeyMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator filter(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getFilterMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator union(com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUnionMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator flatMap(com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getFlatMapMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service ProcessService.
   */
  public static final class ProcessServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<ProcessServiceFutureStub> {
    private ProcessServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected ProcessServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new ProcessServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> map(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getMapMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> mapValues(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getMapValuesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> join(
        com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getJoinMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> mapPartitions(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getMapPartitionsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> glom(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGlomMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> sample(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSampleMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> subtractByKey(
        com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSubtractByKeyMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> filter(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getFilterMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> union(
        com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUnionMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator> flatMap(
        com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getFlatMapMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_MAP = 0;
  private static final int METHODID_MAP_VALUES = 1;
  private static final int METHODID_JOIN = 2;
  private static final int METHODID_REDUCE = 3;
  private static final int METHODID_MAP_PARTITIONS = 4;
  private static final int METHODID_GLOM = 5;
  private static final int METHODID_SAMPLE = 6;
  private static final int METHODID_SUBTRACT_BY_KEY = 7;
  private static final int METHODID_FILTER = 8;
  private static final int METHODID_UNION = 9;
  private static final int METHODID_FLAT_MAP = 10;

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
        case METHODID_MAP:
          serviceImpl.map((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_MAP_VALUES:
          serviceImpl.mapValues((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_JOIN:
          serviceImpl.join((com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_REDUCE:
          serviceImpl.reduce((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.Kv.Operand>) responseObserver);
          break;
        case METHODID_MAP_PARTITIONS:
          serviceImpl.mapPartitions((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_GLOM:
          serviceImpl.glom((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_SAMPLE:
          serviceImpl.sample((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_SUBTRACT_BY_KEY:
          serviceImpl.subtractByKey((com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_FILTER:
          serviceImpl.filter((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_UNION:
          serviceImpl.union((com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
          break;
        case METHODID_FLAT_MAP:
          serviceImpl.flatMap((com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>) responseObserver);
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
          getMapMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_MAP)))
        .addMethod(
          getMapValuesMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_MAP_VALUES)))
        .addMethod(
          getJoinMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_JOIN)))
        .addMethod(
          getReduceMethod(),
          io.grpc.stub.ServerCalls.asyncServerStreamingCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.Kv.Operand>(
                service, METHODID_REDUCE)))
        .addMethod(
          getMapPartitionsMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_MAP_PARTITIONS)))
        .addMethod(
          getGlomMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_GLOM)))
        .addMethod(
          getSampleMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_SAMPLE)))
        .addMethod(
          getSubtractByKeyMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_SUBTRACT_BY_KEY)))
        .addMethod(
          getFilterMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_FILTER)))
        .addMethod(
          getUnionMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.BinaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_UNION)))
        .addMethod(
          getFlatMapMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.computing.processor.Processor.UnaryProcess,
              com.webank.ai.eggroll.api.storage.StorageBasic.StorageLocator>(
                service, METHODID_FLAT_MAP)))
        .build();
  }

  private static abstract class ProcessServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    ProcessServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.eggroll.api.computing.processor.Processor.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("ProcessService");
    }
  }

  private static final class ProcessServiceFileDescriptorSupplier
      extends ProcessServiceBaseDescriptorSupplier {
    ProcessServiceFileDescriptorSupplier() {}
  }

  private static final class ProcessServiceMethodDescriptorSupplier
      extends ProcessServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    ProcessServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (ProcessServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new ProcessServiceFileDescriptorSupplier())
              .addMethod(getMapMethod())
              .addMethod(getMapValuesMethod())
              .addMethod(getJoinMethod())
              .addMethod(getReduceMethod())
              .addMethod(getMapPartitionsMethod())
              .addMethod(getGlomMethod())
              .addMethod(getSampleMethod())
              .addMethod(getSubtractByKeyMethod())
              .addMethod(getFilterMethod())
              .addMethod(getUnionMethod())
              .addMethod(getFlatMapMethod())
              .build();
        }
      }
    }
    return result;
  }
}
