//package com.osx.transfer.grpc;
//
//import com.beust.jcommander.internal.Lists;
//import com.firework.cluster.rpc.FireworkQueueServiceGrpc;
//import com.firework.cluster.rpc.FireworkTransfer;
//
//
//import com.osx.core.constant.Dict;
//import com.osx.core.constant.StatusCode;
//import com.osx.core.constant.TransferStatus;
//import com.osx.core.context.BaseContext;
//import com.osx.core.context.Context;
//import com.osx.core.exceptions.MessageParseException;
//import com.osx.core.service.InboundPackage;
//import com.osx.core.service.OutboundPackage;
//import com.osx.core.utils.JsonUtil;
//import com.osx.transfer.ServiceContainer;
//import com.osx.transfer.queue.TransferQueue;
//import com.osx.transfer.queue.TransferQueueApplyInfo;
//import com.osx.transfer.util.TransferUtil;
//import com.osx.transfer.grpc.stream.ProduceReqStreamObserver;
////import com.firework.transfer.queue.TransferQueueConsumer;
//import com.google.protobuf.ByteString;
//import com.google.protobuf.InvalidProtocolBufferException;
//import io.grpc.stub.StreamObserver;
//
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//
//import java.nio.charset.StandardCharsets;
//import java.util.HashMap;
//import java.util.Map;
//
//import static com.osx.core.constant.StatusCode.SUCCESS;
//import static io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall;
//import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;
//
//public class QueueGrpcService extends FireworkQueueServiceGrpc.FireworkQueueServiceImplBase {
//
//    Logger logger = LoggerFactory.getLogger(QueueGrpcService.class);
//    public StreamObserver<FireworkTransfer.ProduceRequest> produce(
//            StreamObserver<FireworkTransfer.ProduceResponse> responseObserver) {
//        logger.info("receive produce request" );
//        Context context = ContextUtil.buildContext();
//        ProduceReqStreamObserver  produceStreamObserver = new ProduceReqStreamObserver(responseObserver);
//        return produceStreamObserver;
//    }
//
//    /**
//     *  这里应该需要消费者自己维护消费进度，
//     * @param responseObserver
//     * @return
//     */
//    public io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.ConsumeRequest> consume(
//            io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.ConsumeResponse> responseObserver) {
//
//        return new   io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.ConsumeRequest>(){
//
//            io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.ConsumeResponse>  so = responseObserver;
//            @Override
//            public void onNext(FireworkTransfer.ConsumeRequest consumeRequest) {
//                Context context = new BaseContext();
//                long  startOffset = consumeRequest.getStartOffset();
//                String transferId = consumeRequest.getTransferId();
//                FireworkTransfer.ConsumeResponse  consumeResponse;
//                TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(transferId);
//                if(transferQueue!=null){
//                    TransferQueue.TransferQueueConsumeResult transferQueueConsumeResult = transferQueue.consumeOneMessage(context, startOffset);
//                    logger.info("===========TransferQueueConsumeResult {}  {}",transferQueueConsumeResult,startOffset);
//                    if(transferQueue.getTransferStatus()== TransferStatus.TRANSFERING) {
//                        if (transferQueueConsumeResult.getCode() == SUCCESS) {
//                            context.setDataSize(transferQueueConsumeResult.getMessage().getBody().length);
//                            FireworkTransfer.ConsumeResponse.Builder  consumeResponseBuilder = FireworkTransfer.ConsumeResponse.newBuilder();
//                             consumeResponse = TransferUtil.buildResponse(0,"success",transferQueueConsumeResult);
//                        }else{
//                             consumeResponse = TransferUtil.buildResponse(transferQueueConsumeResult.getCode(),"",transferQueueConsumeResult);
//                        }
//                    }else{
//                        consumeResponse = TransferUtil.buildResponse(StatusCode.QUEUE_INVALID_STATUS,"",transferQueueConsumeResult);
//                    }
//                }else {
//                    consumeResponse = TransferUtil.buildResponse(StatusCode.TRANSFER_QUEUE_NOT_FIND, "", null);
//                }
//                so.onNext(consumeResponse);
//            }
//
//            @Override
//            public void onError(Throwable throwable) {
//
//            }
//
//            @Override
//            public void onCompleted() {
//
//            }
//        };
//
//
//
//    }
//
////    /**
////     * 用于polling模式
////     * @param request
////     * @param responseObserver
////     */
////    public void syncQueue(com.firework.cluster.rpc.FireworkTransfer.SyncQueueRequest request,
////                          io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.SyncQueueResponse> responseObserver) {
////        logger.info("receive syncQueueRequest {}",request);
////        ServiceContainer.syncQueueService.sync(request,responseObserver);
////    }
//
//
//
//    public static  FireworkTransfer.SyncQueueResponse   buildResponse(int code , String msgReturn , TransferQueue.TransferQueueConsumeResult messageWraper){
//        FireworkTransfer.SyncQueueResponse.Builder  syncQueueResponseBuilder = FireworkTransfer.SyncQueueResponse.newBuilder();
//        try {
//            syncQueueResponseBuilder.setCode(code);
//            syncQueueResponseBuilder.setMsg(msgReturn);
//            if(messageWraper!=null) {
//                FireworkTransfer.ProduceRequest produceRequest = FireworkTransfer.ProduceRequest.parseFrom(messageWraper.getMessage().getBody());
//                FireworkTransfer.Message msg = produceRequest.getMessage();
//                syncQueueResponseBuilder.setTransferId(produceRequest.getTransferId());
//                syncQueueResponseBuilder.setMessage(msg);
//                syncQueueResponseBuilder.setStartOffset(messageWraper.getRequestIndex());
//                syncQueueResponseBuilder.setTotalOffset(messageWraper.getLogicIndexTotal());
//            }
//        } catch (InvalidProtocolBufferException e) {
//            throw new MessageParseException("message unserialize error");
//        }
//        return syncQueueResponseBuilder.build();
//    }
//
////    public io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.SyncQueueRequest> syncQueue(
////            io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.SyncQueueResponse> responseObserver) {
////        return asyncUnimplementedStreamingCall(getSyncQueueMethod(), responseObserver);
////    }
//
//
//
//
//
//
//
//
//
//
//
//
//    public void queryClusterInfo(com.firework.cluster.rpc.FireworkTransfer.QueryClusterInfoRequest request,
//                                 io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.QueryClusterInfoResponse> responseObserver) {
//        FireworkTransfer.QueryClusterInfoResponse.Builder queryClusterInfoResponseBuilder = FireworkTransfer.QueryClusterInfoResponse.newBuilder();
//        queryClusterInfoResponseBuilder.setCode(SUCCESS);
//        queryClusterInfoResponseBuilder.addAllInstanceIds(Lists.newArrayList(ServiceContainer.transferQueueManager.getInstanceIds()));
//        Map<String, TransferQueueApplyInfo>  globalQueueMap = ServiceContainer.transferQueueManager.getGlobalTransferQueueMap();
//        String  queueData =  JsonUtil.object2Json(globalQueueMap);
//        queryClusterInfoResponseBuilder.setData(ByteString.copyFrom(queueData.getBytes(StandardCharsets.UTF_8)));
//        responseObserver.onNext(queryClusterInfoResponseBuilder.build());
//        responseObserver.onCompleted();
//    }
//
//    public void queryInstanceDetail(com.firework.cluster.rpc.FireworkTransfer.QueryInstanceDetailRequest request,
//                                 io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.QueryInstanceDetailResponse> responseObserver) {
//        FireworkTransfer.QueryInstanceDetailResponse.Builder  queryInstanceDetailResponseBuilder = FireworkTransfer.QueryInstanceDetailResponse.newBuilder();
//        Map<String ,TransferQueue> transferQueueMap =  ServiceContainer.transferQueueManager.getAllLocalQueue();
//        Map queueDataMap =  new HashMap();
//        transferQueueMap.forEach((k,v)->{
//            queueDataMap.put(k,v.getTransferQueueInfo());
//        });
//        String queueData= JsonUtil.object2Json(queueDataMap);
//        queryInstanceDetailResponseBuilder.setCode(SUCCESS);
//        queryInstanceDetailResponseBuilder.setQueueData(ByteString.copyFrom(queueData.getBytes(StandardCharsets.UTF_8)));
//        queryInstanceDetailResponseBuilder.addAllInstanceIds(Lists.newArrayList(ServiceContainer.transferQueueManager.getInstanceIds()));
//        FireworkTransfer.QueryInstanceDetailResponse queryInstanceDetailResponse = queryInstanceDetailResponseBuilder.build();
//        responseObserver.onNext(queryInstanceDetailResponse);
//        responseObserver.onCompleted();
//    }
//
//
//
//
//
//    public void produceUnary(com.firework.cluster.rpc.FireworkTransfer.ProduceRequest request,
//                             io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.ProduceResponse> responseObserver) {
//        //logger.info("produce unary request {}",request);
//        Context  context = ContextUtil.buildContext();
//        InboundPackage<FireworkTransfer.ProduceRequest> data = new InboundPackage<>();
//        data.setBody(request);
//        OutboundPackage<FireworkTransfer.ProduceResponse> outboundPackage= ServiceContainer.producerUnaryService.service(context,data);
//        responseObserver.onNext(outboundPackage.getData());
//        responseObserver.onCompleted();
//    }
//
//    public void ack(com.firework.cluster.rpc.FireworkTransfer.AckRequest request,
//                    io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.AckResponse> responseObserver) {
//        Context context = ContextUtil.buildContext();
//        InboundPackage<FireworkTransfer.AckRequest> data = new InboundPackage<>();
//        data.setBody(request);
//        OutboundPackage<FireworkTransfer.AckResponse> outboundPackage = ServiceContainer.ackService.service(context,data);
//        responseObserver.onNext(outboundPackage.getData());
//        responseObserver.onCompleted();
//    }
//
//
//    public void queryTransferQueueInfo(FireworkTransfer.QueryTransferQueueInfoRequest request,
//                                       io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.QueryTransferQueueInfoResponse> responseObserver) {
//        Context context = ContextUtil.buildContext();
//        InboundPackage<FireworkTransfer.QueryTransferQueueInfoRequest> data = new InboundPackage<>();
//        data.setBody(request);
//        OutboundPackage<FireworkTransfer.QueryTransferQueueInfoResponse> outboundPackage= ServiceContainer.queryTransferQueueService.service(context,data);
//        responseObserver.onNext(outboundPackage.getData());
//        responseObserver.onCompleted();
//
//    }
//
//    public void cancelTransfer(com.firework.cluster.rpc.FireworkTransfer.CancelTransferRequest request,
//                               io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.CancelTransferResponse> responseObserver) {
//
//        Context  context = ContextUtil.buildContext();
//        InboundPackage<FireworkTransfer.CancelTransferRequest> data = new InboundPackage<>();
//        data.setBody(request);
//        OutboundPackage<FireworkTransfer.CancelTransferResponse> outboundPackage =  ServiceContainer.cancelTransferService.service(context,data);
//        responseObserver.onNext(outboundPackage.getData());
//        responseObserver.onCompleted();
//
//    }
//
//    public void consumeUnary(com.firework.cluster.rpc.FireworkTransfer.ConsumeRequest request,
//                             io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.ConsumeResponse> responseObserver) {
//       // logger.info("consumeUnary request {}",request);
//        Context  context = ContextUtil.buildContext();
//        InboundPackage<FireworkTransfer.ConsumeRequest> data = new InboundPackage<>();
//        data.setBody(request);
//
//        context.putData(Dict.RESPONSE_STREAM_OBSERVER,responseObserver);
//        OutboundPackage<FireworkTransfer.ConsumeResponse> outboundPackage = ServiceContainer.consumeUnaryService.service(context,data);
//        //Boolean  needReturnResult = (Boolean)context.getData(Dict.NEED_RETURN_RESULT);
//        if(outboundPackage.getData()!=null) {
//            responseObserver.onNext(outboundPackage.getData());
//            responseObserver.onCompleted();
//        }
//    }
//
//    public void syncTransferInfo(com.firework.cluster.rpc.FireworkTransfer.SyncTransferInfoRequest request,
//                                 io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.SyncTransferInfoResponse> responseObserver) {
//
//        Context  context = ContextUtil.buildContext();
//        InboundPackage<FireworkTransfer.SyncTransferInfoRequest> data = new InboundPackage<>();
//        data.setBody(request);
//
//        context.putData(Dict.RESPONSE_STREAM_OBSERVER,responseObserver);
//        FireworkTransfer.SyncTransferInfoResponse syncTransferInfoResponse = ServiceContainer.transferQueueManager.syncTransferQueueApplyInfo(request);
//        //Boolean  needReturnResult = (Boolean)context.getData(Dict.NEED_RETURN_RESULT);
//            responseObserver.onNext(syncTransferInfoResponse);
//            responseObserver.onCompleted();
//
//    }
//
//
//
//}
