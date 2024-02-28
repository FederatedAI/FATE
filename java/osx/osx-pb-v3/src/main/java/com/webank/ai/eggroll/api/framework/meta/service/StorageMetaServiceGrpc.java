package com.webank.ai.eggroll.api.framework.meta.service;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 * Service to operate storage metadata
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.60.0)",
    comments = "Source: eggroll/meta-service.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class StorageMetaServiceGrpc {

  private StorageMetaServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "com.webank.ai.eggroll.api.framework.meta.service.StorageMetaService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTableMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "createTable",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTableMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTableMethod;
    if ((getCreateTableMethod = StorageMetaServiceGrpc.getCreateTableMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getCreateTableMethod = StorageMetaServiceGrpc.getCreateTableMethod) == null) {
          StorageMetaServiceGrpc.getCreateTableMethod = getCreateTableMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "createTable"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("createTable"))
              .build();
        }
      }
    }
    return getCreateTableMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTableIfAbsentMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "createTableIfAbsent",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTableIfAbsentMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateTableIfAbsentMethod;
    if ((getCreateTableIfAbsentMethod = StorageMetaServiceGrpc.getCreateTableIfAbsentMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getCreateTableIfAbsentMethod = StorageMetaServiceGrpc.getCreateTableIfAbsentMethod) == null) {
          StorageMetaServiceGrpc.getCreateTableIfAbsentMethod = getCreateTableIfAbsentMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "createTableIfAbsent"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("createTableIfAbsent"))
              .build();
        }
      }
    }
    return getCreateTableIfAbsentMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateTableMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "updateTable",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateTableMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateTableMethod;
    if ((getUpdateTableMethod = StorageMetaServiceGrpc.getUpdateTableMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getUpdateTableMethod = StorageMetaServiceGrpc.getUpdateTableMethod) == null) {
          StorageMetaServiceGrpc.getUpdateTableMethod = getUpdateTableMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "updateTable"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("updateTable"))
              .build();
        }
      }
    }
    return getUpdateTableMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateFragmentsForTableMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "createFragmentsForTable",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateFragmentsForTableMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getCreateFragmentsForTableMethod;
    if ((getCreateFragmentsForTableMethod = StorageMetaServiceGrpc.getCreateFragmentsForTableMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getCreateFragmentsForTableMethod = StorageMetaServiceGrpc.getCreateFragmentsForTableMethod) == null) {
          StorageMetaServiceGrpc.getCreateFragmentsForTableMethod = getCreateFragmentsForTableMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "createFragmentsForTable"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("createFragmentsForTable"))
              .build();
        }
      }
    }
    return getCreateFragmentsForTableMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateFragmentMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "updateFragment",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateFragmentMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getUpdateFragmentMethod;
    if ((getUpdateFragmentMethod = StorageMetaServiceGrpc.getUpdateFragmentMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getUpdateFragmentMethod = StorageMetaServiceGrpc.getUpdateFragmentMethod) == null) {
          StorageMetaServiceGrpc.getUpdateFragmentMethod = getUpdateFragmentMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "updateFragment"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("updateFragment"))
              .build();
        }
      }
    }
    return getUpdateFragmentMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTableByIdMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getTableById",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTableByIdMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTableByIdMethod;
    if ((getGetTableByIdMethod = StorageMetaServiceGrpc.getGetTableByIdMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetTableByIdMethod = StorageMetaServiceGrpc.getGetTableByIdMethod) == null) {
          StorageMetaServiceGrpc.getGetTableByIdMethod = getGetTableByIdMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getTableById"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getTableById"))
              .build();
        }
      }
    }
    return getGetTableByIdMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTableMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getTable",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTableMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTableMethod;
    if ((getGetTableMethod = StorageMetaServiceGrpc.getGetTableMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetTableMethod = StorageMetaServiceGrpc.getGetTableMethod) == null) {
          StorageMetaServiceGrpc.getGetTableMethod = getGetTableMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getTable"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getTable"))
              .build();
        }
      }
    }
    return getGetTableMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTablesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getTables",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTablesMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetTablesMethod;
    if ((getGetTablesMethod = StorageMetaServiceGrpc.getGetTablesMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetTablesMethod = StorageMetaServiceGrpc.getGetTablesMethod) == null) {
          StorageMetaServiceGrpc.getGetTablesMethod = getGetTablesMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getTables"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getTables"))
              .build();
        }
      }
    }
    return getGetTablesMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetFragmentByIdMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getFragmentById",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetFragmentByIdMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetFragmentByIdMethod;
    if ((getGetFragmentByIdMethod = StorageMetaServiceGrpc.getGetFragmentByIdMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetFragmentByIdMethod = StorageMetaServiceGrpc.getGetFragmentByIdMethod) == null) {
          StorageMetaServiceGrpc.getGetFragmentByIdMethod = getGetFragmentByIdMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getFragmentById"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getFragmentById"))
              .build();
        }
      }
    }
    return getGetFragmentByIdMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetFragmentsByTableIdMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getFragmentsByTableId",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetFragmentsByTableIdMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetFragmentsByTableIdMethod;
    if ((getGetFragmentsByTableIdMethod = StorageMetaServiceGrpc.getGetFragmentsByTableIdMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetFragmentsByTableIdMethod = StorageMetaServiceGrpc.getGetFragmentsByTableIdMethod) == null) {
          StorageMetaServiceGrpc.getGetFragmentsByTableIdMethod = getGetFragmentsByTableIdMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getFragmentsByTableId"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getFragmentsByTableId"))
              .build();
        }
      }
    }
    return getGetFragmentsByTableIdMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetStorageNodesByTableIdMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getStorageNodesByTableId",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetStorageNodesByTableIdMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetStorageNodesByTableIdMethod;
    if ((getGetStorageNodesByTableIdMethod = StorageMetaServiceGrpc.getGetStorageNodesByTableIdMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetStorageNodesByTableIdMethod = StorageMetaServiceGrpc.getGetStorageNodesByTableIdMethod) == null) {
          StorageMetaServiceGrpc.getGetStorageNodesByTableIdMethod = getGetStorageNodesByTableIdMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getStorageNodesByTableId"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getStorageNodesByTableId"))
              .build();
        }
      }
    }
    return getGetStorageNodesByTableIdMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetEggNodeManagerByIpMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getEggNodeManagerByIp",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetEggNodeManagerByIpMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetEggNodeManagerByIpMethod;
    if ((getGetEggNodeManagerByIpMethod = StorageMetaServiceGrpc.getGetEggNodeManagerByIpMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetEggNodeManagerByIpMethod = StorageMetaServiceGrpc.getGetEggNodeManagerByIpMethod) == null) {
          StorageMetaServiceGrpc.getGetEggNodeManagerByIpMethod = getGetEggNodeManagerByIpMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getEggNodeManagerByIp"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getEggNodeManagerByIp"))
              .build();
        }
      }
    }
    return getGetEggNodeManagerByIpMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodeByIdMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getNodeById",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodeByIdMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodeByIdMethod;
    if ((getGetNodeByIdMethod = StorageMetaServiceGrpc.getGetNodeByIdMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetNodeByIdMethod = StorageMetaServiceGrpc.getGetNodeByIdMethod) == null) {
          StorageMetaServiceGrpc.getGetNodeByIdMethod = getGetNodeByIdMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getNodeById"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getNodeById"))
              .build();
        }
      }
    }
    return getGetNodeByIdMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesByIdsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getNodesByIds",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesByIdsMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesByIdsMethod;
    if ((getGetNodesByIdsMethod = StorageMetaServiceGrpc.getGetNodesByIdsMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetNodesByIdsMethod = StorageMetaServiceGrpc.getGetNodesByIdsMethod) == null) {
          StorageMetaServiceGrpc.getGetNodesByIdsMethod = getGetNodesByIdsMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getNodesByIds"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getNodesByIds"))
              .build();
        }
      }
    }
    return getGetNodesByIdsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesOfStatusMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getNodesOfStatus",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesOfStatusMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesOfStatusMethod;
    if ((getGetNodesOfStatusMethod = StorageMetaServiceGrpc.getGetNodesOfStatusMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetNodesOfStatusMethod = StorageMetaServiceGrpc.getGetNodesOfStatusMethod) == null) {
          StorageMetaServiceGrpc.getGetNodesOfStatusMethod = getGetNodesOfStatusMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getNodesOfStatus"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getNodesOfStatus"))
              .build();
        }
      }
    }
    return getGetNodesOfStatusMethod;
  }

  private static volatile io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "getNodes",
      requestType = com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.class,
      responseType = com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
      com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesMethod() {
    io.grpc.MethodDescriptor<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getGetNodesMethod;
    if ((getGetNodesMethod = StorageMetaServiceGrpc.getGetNodesMethod) == null) {
      synchronized (StorageMetaServiceGrpc.class) {
        if ((getGetNodesMethod = StorageMetaServiceGrpc.getGetNodesMethod) == null) {
          StorageMetaServiceGrpc.getGetNodesMethod = getGetNodesMethod =
              io.grpc.MethodDescriptor.<com.webank.ai.eggroll.api.core.BasicMeta.CallRequest, com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "getNodes"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  com.webank.ai.eggroll.api.core.BasicMeta.CallResponse.getDefaultInstance()))
              .setSchemaDescriptor(new StorageMetaServiceMethodDescriptorSupplier("getNodes"))
              .build();
        }
      }
    }
    return getGetNodesMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static StorageMetaServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<StorageMetaServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<StorageMetaServiceStub>() {
        @java.lang.Override
        public StorageMetaServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new StorageMetaServiceStub(channel, callOptions);
        }
      };
    return StorageMetaServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static StorageMetaServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<StorageMetaServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<StorageMetaServiceBlockingStub>() {
        @java.lang.Override
        public StorageMetaServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new StorageMetaServiceBlockingStub(channel, callOptions);
        }
      };
    return StorageMetaServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static StorageMetaServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<StorageMetaServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<StorageMetaServiceFutureStub>() {
        @java.lang.Override
        public StorageMetaServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new StorageMetaServiceFutureStub(channel, callOptions);
        }
      };
    return StorageMetaServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   * Service to operate storage metadata
   * </pre>
   */
  public interface AsyncService {

    /**
     */
    default void createTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCreateTableMethod(), responseObserver);
    }

    /**
     */
    default void createTableIfAbsent(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCreateTableIfAbsentMethod(), responseObserver);
    }

    /**
     */
    default void updateTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUpdateTableMethod(), responseObserver);
    }

    /**
     */
    default void createFragmentsForTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCreateFragmentsForTableMethod(), responseObserver);
    }

    /**
     */
    default void updateFragment(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getUpdateFragmentMethod(), responseObserver);
    }

    /**
     */
    default void getTableById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetTableByIdMethod(), responseObserver);
    }

    /**
     */
    default void getTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetTableMethod(), responseObserver);
    }

    /**
     */
    default void getTables(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetTablesMethod(), responseObserver);
    }

    /**
     */
    default void getFragmentById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetFragmentByIdMethod(), responseObserver);
    }

    /**
     */
    default void getFragmentsByTableId(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetFragmentsByTableIdMethod(), responseObserver);
    }

    /**
     */
    default void getStorageNodesByTableId(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetStorageNodesByTableIdMethod(), responseObserver);
    }

    /**
     */
    default void getEggNodeManagerByIp(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetEggNodeManagerByIpMethod(), responseObserver);
    }

    /**
     */
    default void getNodeById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetNodeByIdMethod(), responseObserver);
    }

    /**
     */
    default void getNodesByIds(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetNodesByIdsMethod(), responseObserver);
    }

    /**
     */
    default void getNodesOfStatus(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetNodesOfStatusMethod(), responseObserver);
    }

    /**
     */
    default void getNodes(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getGetNodesMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service StorageMetaService.
   * <pre>
   * Service to operate storage metadata
   * </pre>
   */
  public static abstract class StorageMetaServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return StorageMetaServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service StorageMetaService.
   * <pre>
   * Service to operate storage metadata
   * </pre>
   */
  public static final class StorageMetaServiceStub
      extends io.grpc.stub.AbstractAsyncStub<StorageMetaServiceStub> {
    private StorageMetaServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected StorageMetaServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new StorageMetaServiceStub(channel, callOptions);
    }

    /**
     */
    public void createTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCreateTableMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void createTableIfAbsent(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCreateTableIfAbsentMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void updateTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUpdateTableMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void createFragmentsForTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCreateFragmentsForTableMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void updateFragment(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getUpdateFragmentMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getTableById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetTableByIdMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetTableMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getTables(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetTablesMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getFragmentById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetFragmentByIdMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getFragmentsByTableId(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetFragmentsByTableIdMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getStorageNodesByTableId(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetStorageNodesByTableIdMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getEggNodeManagerByIp(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetEggNodeManagerByIpMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getNodeById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetNodeByIdMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getNodesByIds(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetNodesByIdsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getNodesOfStatus(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetNodesOfStatusMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     */
    public void getNodes(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request,
        io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getGetNodesMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service StorageMetaService.
   * <pre>
   * Service to operate storage metadata
   * </pre>
   */
  public static final class StorageMetaServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<StorageMetaServiceBlockingStub> {
    private StorageMetaServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected StorageMetaServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new StorageMetaServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse createTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCreateTableMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse createTableIfAbsent(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCreateTableIfAbsentMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse updateTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUpdateTableMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse createFragmentsForTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCreateFragmentsForTableMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse updateFragment(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getUpdateFragmentMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getTableById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetTableByIdMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getTable(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetTableMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getTables(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetTablesMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getFragmentById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetFragmentByIdMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getFragmentsByTableId(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetFragmentsByTableIdMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getStorageNodesByTableId(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetStorageNodesByTableIdMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getEggNodeManagerByIp(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetEggNodeManagerByIpMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getNodeById(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetNodeByIdMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getNodesByIds(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetNodesByIdsMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getNodesOfStatus(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetNodesOfStatusMethod(), getCallOptions(), request);
    }

    /**
     */
    public com.webank.ai.eggroll.api.core.BasicMeta.CallResponse getNodes(com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getGetNodesMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service StorageMetaService.
   * <pre>
   * Service to operate storage metadata
   * </pre>
   */
  public static final class StorageMetaServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<StorageMetaServiceFutureStub> {
    private StorageMetaServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected StorageMetaServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new StorageMetaServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> createTable(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCreateTableMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> createTableIfAbsent(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCreateTableIfAbsentMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> updateTable(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUpdateTableMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> createFragmentsForTable(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCreateFragmentsForTableMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> updateFragment(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getUpdateFragmentMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getTableById(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetTableByIdMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getTable(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetTableMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getTables(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetTablesMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getFragmentById(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetFragmentByIdMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getFragmentsByTableId(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetFragmentsByTableIdMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getStorageNodesByTableId(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetStorageNodesByTableIdMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getEggNodeManagerByIp(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetEggNodeManagerByIpMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getNodeById(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetNodeByIdMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getNodesByIds(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetNodesByIdsMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getNodesOfStatus(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetNodesOfStatusMethod(), getCallOptions()), request);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse> getNodes(
        com.webank.ai.eggroll.api.core.BasicMeta.CallRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getGetNodesMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_CREATE_TABLE = 0;
  private static final int METHODID_CREATE_TABLE_IF_ABSENT = 1;
  private static final int METHODID_UPDATE_TABLE = 2;
  private static final int METHODID_CREATE_FRAGMENTS_FOR_TABLE = 3;
  private static final int METHODID_UPDATE_FRAGMENT = 4;
  private static final int METHODID_GET_TABLE_BY_ID = 5;
  private static final int METHODID_GET_TABLE = 6;
  private static final int METHODID_GET_TABLES = 7;
  private static final int METHODID_GET_FRAGMENT_BY_ID = 8;
  private static final int METHODID_GET_FRAGMENTS_BY_TABLE_ID = 9;
  private static final int METHODID_GET_STORAGE_NODES_BY_TABLE_ID = 10;
  private static final int METHODID_GET_EGG_NODE_MANAGER_BY_IP = 11;
  private static final int METHODID_GET_NODE_BY_ID = 12;
  private static final int METHODID_GET_NODES_BY_IDS = 13;
  private static final int METHODID_GET_NODES_OF_STATUS = 14;
  private static final int METHODID_GET_NODES = 15;

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
        case METHODID_CREATE_TABLE:
          serviceImpl.createTable((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_CREATE_TABLE_IF_ABSENT:
          serviceImpl.createTableIfAbsent((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_UPDATE_TABLE:
          serviceImpl.updateTable((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_CREATE_FRAGMENTS_FOR_TABLE:
          serviceImpl.createFragmentsForTable((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_UPDATE_FRAGMENT:
          serviceImpl.updateFragment((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_TABLE_BY_ID:
          serviceImpl.getTableById((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_TABLE:
          serviceImpl.getTable((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_TABLES:
          serviceImpl.getTables((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_FRAGMENT_BY_ID:
          serviceImpl.getFragmentById((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_FRAGMENTS_BY_TABLE_ID:
          serviceImpl.getFragmentsByTableId((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_STORAGE_NODES_BY_TABLE_ID:
          serviceImpl.getStorageNodesByTableId((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_EGG_NODE_MANAGER_BY_IP:
          serviceImpl.getEggNodeManagerByIp((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_NODE_BY_ID:
          serviceImpl.getNodeById((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_NODES_BY_IDS:
          serviceImpl.getNodesByIds((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_NODES_OF_STATUS:
          serviceImpl.getNodesOfStatus((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
              (io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>) responseObserver);
          break;
        case METHODID_GET_NODES:
          serviceImpl.getNodes((com.webank.ai.eggroll.api.core.BasicMeta.CallRequest) request,
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
          getCreateTableMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_CREATE_TABLE)))
        .addMethod(
          getCreateTableIfAbsentMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_CREATE_TABLE_IF_ABSENT)))
        .addMethod(
          getUpdateTableMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_UPDATE_TABLE)))
        .addMethod(
          getCreateFragmentsForTableMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_CREATE_FRAGMENTS_FOR_TABLE)))
        .addMethod(
          getUpdateFragmentMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_UPDATE_FRAGMENT)))
        .addMethod(
          getGetTableByIdMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_TABLE_BY_ID)))
        .addMethod(
          getGetTableMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_TABLE)))
        .addMethod(
          getGetTablesMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_TABLES)))
        .addMethod(
          getGetFragmentByIdMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_FRAGMENT_BY_ID)))
        .addMethod(
          getGetFragmentsByTableIdMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_FRAGMENTS_BY_TABLE_ID)))
        .addMethod(
          getGetStorageNodesByTableIdMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_STORAGE_NODES_BY_TABLE_ID)))
        .addMethod(
          getGetEggNodeManagerByIpMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_EGG_NODE_MANAGER_BY_IP)))
        .addMethod(
          getGetNodeByIdMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_NODE_BY_ID)))
        .addMethod(
          getGetNodesByIdsMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_NODES_BY_IDS)))
        .addMethod(
          getGetNodesOfStatusMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_NODES_OF_STATUS)))
        .addMethod(
          getGetNodesMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              com.webank.ai.eggroll.api.core.BasicMeta.CallRequest,
              com.webank.ai.eggroll.api.core.BasicMeta.CallResponse>(
                service, METHODID_GET_NODES)))
        .build();
  }

  private static abstract class StorageMetaServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    StorageMetaServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return com.webank.ai.eggroll.api.framework.meta.service.MetaService.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("StorageMetaService");
    }
  }

  private static final class StorageMetaServiceFileDescriptorSupplier
      extends StorageMetaServiceBaseDescriptorSupplier {
    StorageMetaServiceFileDescriptorSupplier() {}
  }

  private static final class StorageMetaServiceMethodDescriptorSupplier
      extends StorageMetaServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    StorageMetaServiceMethodDescriptorSupplier(java.lang.String methodName) {
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
      synchronized (StorageMetaServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new StorageMetaServiceFileDescriptorSupplier())
              .addMethod(getCreateTableMethod())
              .addMethod(getCreateTableIfAbsentMethod())
              .addMethod(getUpdateTableMethod())
              .addMethod(getCreateFragmentsForTableMethod())
              .addMethod(getUpdateFragmentMethod())
              .addMethod(getGetTableByIdMethod())
              .addMethod(getGetTableMethod())
              .addMethod(getGetTablesMethod())
              .addMethod(getGetFragmentByIdMethod())
              .addMethod(getGetFragmentsByTableIdMethod())
              .addMethod(getGetStorageNodesByTableIdMethod())
              .addMethod(getGetEggNodeManagerByIpMethod())
              .addMethod(getGetNodeByIdMethod())
              .addMethod(getGetNodesByIdsMethod())
              .addMethod(getGetNodesOfStatusMethod())
              .addMethod(getGetNodesMethod())
              .build();
        }
      }
    }
    return result;
  }
}
