package org.fedai.osx.broker.test.grpc;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class OldFateTest {

    static int port = 9370;//9371
   static String ip = "localhost";


    static Logger logger = LoggerFactory.getLogger(OldFateTest.class);

    static boolean useSSL = false;

    public static ManagedChannel createManagedChannel(String ip, int port) {
        try {
            NettyChannelBuilder channelBuilder = NettyChannelBuilder
                    .forAddress(ip, port)
                    .keepAliveTime(60, TimeUnit.SECONDS)
                    .keepAliveTimeout(60, TimeUnit.SECONDS)
                    .keepAliveWithoutCalls(true)
                    .idleTimeout(60, TimeUnit.SECONDS)
                    .perRpcBufferLimit(128 << 20)
                    .flowControlWindow(32 << 20)
                    .maxInboundMessageSize(32 << 20)
                    .enableRetry()
                    .retryBufferSize(16 << 20)
                    .maxRetryAttempts(20);

            if (useSSL) {
                SslContextBuilder sslContextBuilder = GrpcSslContexts.forClient()
                        .keyManager(new File("/Users/kaideng/work/cert/cjt.crt"),
                                new File("/Users/kaideng/work/cert/cjt.key"))
                        .trustManager(new File("/Users/kaideng/work/cert/FATE-Cloud.crt"))
//                        .keyManager(new File("/Users/kaideng/work/cert/yl/yl.crt"),
//                                new File("/Users/kaideng/work/cert/yl/yl.key"))
//                        .trustManager(new File("/Users/kaideng/work/cert/yl/fdn-ca.crt"))


                        .sessionTimeout(3600 << 4)
                        .sessionCacheSize(65536);
                channelBuilder.sslContext(sslContextBuilder.build()).useTransportSecurity();

//                logger.info("running in secure mode for endpoint {}:{}, client crt path: {}, client key path: {}, ca crt path: {}.",
//                        ip, port, nettyServerInfo.getCertChainFilePath(), nettyServerInfo.getPrivateKeyFilePath(),
//                        nettyServerInfo.getTrustCertCollectionFilePath());
            } else {
                channelBuilder.usePlaintext();
            }
            return channelBuilder.build();
        } catch (Exception e) {
            e.printStackTrace();
            //      logger.error("create channel error : " ,e);
            //e.printStackTrace();
        }
        return null;
    }

    public static void testUnaryCall() {
        logger.info("test unary call");
        ManagedChannel managedChannel = createManagedChannel(ip, port);
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub = DataTransferServiceGrpc.newBlockingStub(managedChannel);
        Proxy.Packet.Builder builder = Proxy.Packet.newBuilder();
        Transfer.RollSiteHeader.Builder headerBuilder = Transfer.RollSiteHeader.newBuilder();
        headerBuilder.setDstPartyId("10001");
        builder.setHeader(Proxy.Metadata.newBuilder().setExt(headerBuilder.build().toByteString()));
        Proxy.Data.Builder dataBuilder = Proxy.Data.newBuilder();
        dataBuilder.setKey("name");
        dataBuilder.setValue(ByteString.copyFrom(("xiaoxiao").getBytes()));
        builder.setBody(Proxy.Data.newBuilder().setValue(ByteString.copyFromUtf8("kaideng")));
        Proxy.Packet result = stub.unaryCall(builder.build());
        System.err.println(result);
    }


//    public static void  testPull(){
//        new Thread(()-> {
//            ManagedChannel managedChannel = createManagedChannel("localhost", port);
//            DataTransferServiceGrpc.DataTransferServiceStub stub = DataTransferServiceGrpc.newStub(managedChannel);
//            StreamObserver<Proxy.Packet> responseOb = new StreamObserver<Proxy.Packet>() {
//                @Override
//                public void onNext(Proxy.Packet value) {
//                    System.err.println("response onNext");
//                }
//
//                @Override
//                public void onError(Throwable t) {
//                    t.printStackTrace();
//                    System.err.println("response onError");
//                    logger.error("===========error ",t);
//                }
//
//                @Override
//                public void onCompleted() {
//                    System.err.println("response onCompleted");
//                }
//            };
//            Proxy.Metadata metadata = Proxy.Metadata.newBuilder().build();
//            stub.pull(metadata, responseOb);
//        }).start();
//    }


    public static void testPush() {

        ManagedChannel managedChannel = createManagedChannel("localhost", port);

        DataTransferServiceGrpc.DataTransferServiceStub stub = DataTransferServiceGrpc.newStub(managedChannel);

        StreamObserver<Proxy.Metadata> responseOb = new StreamObserver<Proxy.Metadata>() {
            @Override
            public void onNext(Proxy.Metadata value) {
                System.err.println("response onNext");
            }

            @Override
            public void onError(Throwable t) {
                logger.error("on Error {}", t.getMessage());
                t.printStackTrace();
            }

            @Override
            public void onCompleted() {
                System.err.println("response onCompleted");
            }
        };

        //while(true)
        {
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            for (int t = 0; t < 1; t++) {

        String  srcPartyId =  "9999";
        String  desPartyId =  "10000";

//                new Thread(() -> {
            StreamObserver<Proxy.Packet> requestOb = stub.push(responseOb);
            for (int i = 0; i < 3; i++) {

//                Proxy.Metadata metadata = packet.getHeader();
//                ByteString encodedRollSiteHeader = metadata.getExt();
                Transfer.RollSiteHeader.Builder  rollSiteHeader = Transfer.RollSiteHeader.newBuilder();
                rollSiteHeader.setDstRole("desRole");
                rollSiteHeader.setDstPartyId(desPartyId);
                rollSiteHeader.setSrcPartyId(srcPartyId);
                rollSiteHeader.setSrcRole("srcRole");
                Proxy.Packet.Builder packetBuilder = Proxy.Packet.newBuilder();
                packetBuilder.setHeader(Proxy.Metadata.newBuilder().setSrc(Proxy.Topic.newBuilder().setPartyId("10000"))
                        .setDst(Proxy.Topic.newBuilder().setPartyId("9999").setName("kaidengTestTopic").build())
                                .setExt(rollSiteHeader.build().toByteString())
                        .build());
//                Transfer.RollSiteHeader.Builder headerBuilder = Transfer.RollSiteHeader.newBuilder();
//                headerBuilder.setDstPartyId("10000");
                //   packetBuilder.setHeader(Proxy.Metadata.newBuilder().setExt(headerBuilder.build().toByteString()));
                Proxy.Data.Builder dataBuilder = Proxy.Data.newBuilder();
                dataBuilder.setKey("name");
                dataBuilder.setValue(ByteString.copyFrom(("xiaoxiao" + i).getBytes()));
                packetBuilder.setBody(dataBuilder.build());

                if (i == 99) {
                    //     throw  new RuntimeException();
                }
                requestOb.onNext(packetBuilder.build());
                System.err.println("test send !!!!!!!!!!!!!!!!!!!!!!");
            }
            requestOb.onCompleted();


//                }).start();
            //      }
        }

    }


    public static void main(String[] args) {
        System.err.println("===============");
        testPush();
        //testUnaryCall();
        CountDownLatch countDownLatch = new CountDownLatch(1);
        try {
            countDownLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

}
