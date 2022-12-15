//package com.osx.transfer.grpc;
//
//import com.firework.cluster.rpc.FireworkTransfer;
//import com.osx.core.context.Context;
//import com.osx.transfer.consumer.ConsumerManager;
//import com.osx.transfer.consumer.StreamConsumer;
//import com.osx.transfer.queue.TransferQueue;
//import com.osx.transfer.queue.TransferQueueManager;
//import io.grpc.stub.StreamObserver;
//
//public class ConsumeRequestStreamObserver implements StreamObserver<FireworkTransfer.ConsumeRequest> {
//
//
//    TransferQueueManager transferQueueManager;
//
//    ConsumerManager consumerManager;
//
//    Context context;
//
//    StreamObserver<FireworkTransfer.ConsumeResponse> responseSb;
//
//
//
//
//    @Override
//    public void onNext(FireworkTransfer.ConsumeRequest consumeRequest) {
//
//      String transferId = consumeRequest.getTransferId();
//      long  startOffset = consumeRequest.getStartOffset();
//      //集群逻辑还未处理
//
//      StreamConsumer streamConsumer =  consumerManager.getOrCreateStreamConsumer(transferId);
//      TransferQueue.TransferQueueConsumeResult consumeResult = streamConsumer.consume(context ,startOffset);
//      FireworkTransfer.ConsumeResponse  consumeResponse =  buildConsumeResponse();
//
//      responseSb.onNext(consumeResponse);
//
//    }
//
//    @Override
//    public void onError(Throwable throwable) {
//        responseSb.onError(throwable);
//    }
//
//    @Override
//    public void onCompleted() {
//        responseSb.onCompleted();
//    }
//
//
//    FireworkTransfer.ConsumeResponse buildConsumeResponse(){
//
//
//        return  null;
//    }
//}
