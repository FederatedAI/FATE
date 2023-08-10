//package com.firework.transfer.grpc;
//
//import com.firework.cluster.rpc.FireworkQueueServiceGrpc;
//import com.firework.cluster.rpc.FireworkTransfer;
//import com.google.protobuf.ByteString;
//import io.grpc.ManagedChannel;
//import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
//import io.grpc.stub.StreamObserver;
//import org.junit.Before;
//import org.junit.Test;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.util.UUID;
//import java.util.concurrent.CountDownLatch;
//import java.util.concurrent.TimeUnit;
//
//import static org.junit.Assert.assertNotNull;
//import static org.junit.Assert.assertTrue;
//
//public class SyncQueueTest {
//
//    Logger logger = LoggerFactory.getLogger(SyncQueueTest.class);
//    String  ip=  "localhost";
//    //int port = 8250;//nginx
//    int port = 9889;//nginx
//    String desPartyId = "10000";
//    String desRole = "";
//    String srcPartyId = "9999";
//    String srcRole = "";
//    String transferId = "testTransferId02";
//    String sessionId =  "testSessionId";
//
//    FireworkQueueServiceGrpc.FireworkQueueServiceStub   stub;
//    FireworkQueueServiceGrpc.FireworkQueueServiceBlockingStub  blockingStub;
//
//    @Before
//    public  void init (){
//        ManagedChannel managedChannel = createManagedChannel(ip,port);
//        stub =  FireworkQueueServiceGrpc.newStub(managedChannel);
//        ManagedChannel managedChannel2 = createManagedChannel(ip,port);
//        blockingStub = FireworkQueueServiceGrpc.newBlockingStub(managedChannel2);
//    }
//
//
//
//
////
////    @Test
////    public  void  testSyncQueue(){
////
////        CountDownLatch  countDownLatch = new CountDownLatch(1);
////
////        FireworkTransfer.SyncQueueRequest.Builder builder = FireworkTransfer.SyncQueueRequest.newBuilder();
////        builder.setTransferId(this.transferId);
////        FireworkTransfer.SyncQueueRequest syncQueueRequest = builder.build();
////        StreamObserver<FireworkTransfer.SyncQueueResponse> streamObserver = new   StreamObserver<FireworkTransfer.SyncQueueResponse>(){
////            @Override
////            public void onNext(FireworkTransfer.SyncQueueResponse syncQueueResponse) {
////                logger.info("=========== syncQueueResponse : {}",syncQueueResponse);
////            }
////            @Override
////            public void onError(Throwable throwable) {
////                logger.error("=========== syncQueueResponse error",throwable);
////                countDownLatch.countDown();
////            }
////            @Override
////            public void onCompleted() {
////                logger.error("=========== syncQueueResponse complete");
////                countDownLatch.countDown();
////            }
////        };
////        stub.syncQueue(syncQueueRequest, streamObserver);
////
////        try {
////            countDownLatch.await();
////        } catch (InterruptedException e) {
////            e.printStackTrace();
////        }
////        logger.info("=============================");
////    }
//
//
//
//
//
//
//    @Test
//    public  void test02Query() {
//        System.err.println("testQuery");
//        FireworkTransfer.QueryTransferQueueInfoRequest.Builder builder = com.firework.cluster.rpc.FireworkTransfer.QueryTransferQueueInfoRequest.newBuilder();
//        builder.setTransferId(this.transferId);
//        builder.setSessionId(sessionId);
//        FireworkTransfer.QueryTransferQueueInfoRequest queryTopicRequest = builder.build();
//        FireworkTransfer.QueryTransferQueueInfoResponse queryTopicResponse = blockingStub.queryTransferQueueInfo(queryTopicRequest);
//        System.err.println(queryTopicResponse);
//        assertNotNull(queryTopicResponse);
//        assertTrue(queryTopicResponse.getCode() == 0);
//
//    }
//
//
//
//
//
//
//
//    @Test
//    public  void  test04UnaryProduce(){
//
//        for (int i = 0; i < 10; i++) {
//            FireworkTransfer.ProduceRequest.Builder  produceRequestBuilder = FireworkTransfer.ProduceRequest.newBuilder();
//            FireworkTransfer.RouteInfo.Builder  routerInfoBuilder = FireworkTransfer.RouteInfo.newBuilder();
//            routerInfoBuilder.setDesPartyId(desPartyId);
//            routerInfoBuilder.setDesRole(desRole);
//            routerInfoBuilder.setSrcPartyId(srcPartyId);
//            routerInfoBuilder.setSrcRole(srcRole);
//            produceRequestBuilder.setRouteInfo(routerInfoBuilder.build());
//            produceRequestBuilder.setTransferId(transferId);
//            produceRequestBuilder.setSessionId(UUID.randomUUID().toString());
//            FireworkTransfer.Message.Builder  messageBuilder = FireworkTransfer.Message.newBuilder();
//            messageBuilder.setBody(ByteString.copyFrom(("test body element "+i).getBytes()));
//            messageBuilder.setHead(ByteString.copyFrom(("test head "+i).getBytes()));
//            produceRequestBuilder.setMessage(messageBuilder.build());
//            FireworkTransfer.ProduceResponse produceResponse = blockingStub.produceUnary(produceRequestBuilder.build());
//            System.err.println("response " +produceResponse);
//            try {
//                Thread.sleep(100);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//        }
//    }
//
//
//
//
//
//
//
//
//
//    @Test
//    public  void  test07CancelTransfer(){
//        FireworkTransfer.CancelTransferRequest     cancelTransferRequest =  FireworkTransfer.CancelTransferRequest.newBuilder()
//                .setTransferId(transferId).setSessionId(sessionId).build();
//        FireworkTransfer.CancelTransferResponse cancelTransferResponse  = blockingStub.cancelTransfer(cancelTransferRequest);
//        System.err.println(cancelTransferResponse);
//
//    }
//
//
//    public static ManagedChannel createManagedChannel(String ip, int port) {
//        try {
//            NettyChannelBuilder channelBuilder = NettyChannelBuilder
//                    .forAddress(ip, port)
//                    .keepAliveTime(60, TimeUnit.SECONDS)
//                    .keepAliveTimeout(60, TimeUnit.SECONDS)
//                    .keepAliveWithoutCalls(true)
//                    .idleTimeout(60, TimeUnit.SECONDS)
//                    .perRpcBufferLimit(128 << 20)
//                    .flowControlWindow(32 << 20)
//                    .maxInboundMessageSize(32 << 20)
//                    .enableRetry()
//                    .retryBufferSize(16 << 20)
//                    .maxRetryAttempts(20);
//
//            channelBuilder.usePlaintext();
//            return channelBuilder.build();
//        }
//        catch (Exception e) {
//            e.printStackTrace();
//        }
//        return null;
//    }
//
//}
