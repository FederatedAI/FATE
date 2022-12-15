//package com.firework.transfer.token;
//
//import com.firework.cluster.rpc.Firework;
//import com.firework.cluster.rpc.FireworkServiceGrpc;
//import io.grpc.ManagedChannel;
//import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
//
//import java.util.concurrent.TimeUnit;
//
//public class TokenClientTest {
//
//    public  static  void  main(String[] args){
//        NettyChannelBuilder channelBuilder = NettyChannelBuilder
//                .forAddress("localhost", 9678)
//                .keepAliveTime(60, TimeUnit.SECONDS)
//                .keepAliveTimeout(60, TimeUnit.SECONDS)
//                .keepAliveWithoutCalls(true)
//                .idleTimeout(60, TimeUnit.SECONDS)
//                .perRpcBufferLimit(128 << 20)
//                .flowControlWindow(32 << 20)
//                .maxInboundMessageSize(32 << 20)
//                .enableRetry()
//                .retryBufferSize(16 << 20)
//                .maxRetryAttempts(20);
////        NettyChannelBuilder  nettyChannelBuilder = new NettyChannelBuilder("localhost",9678);
//        channelBuilder.usePlaintext();
//        FireworkServiceGrpc.FireworkServiceBlockingStub  blockingStub = FireworkServiceGrpc.newBlockingStub(channelBuilder.build());
//
//        Firework.TokenRequest.Builder tokenRequestBuilder = Firework.TokenRequest.newBuilder();
//        tokenRequestBuilder.setCount(10);
//        tokenRequestBuilder.setResource("test");
//        Firework.TokenResponse tokenResponse = blockingStub.applyToken(tokenRequestBuilder.build());
//        System.err.println("===============  token response ========="+tokenResponse);
//
//        System.err.println("over");
//    }
//}
