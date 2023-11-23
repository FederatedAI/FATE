package org.fedai.osx.broker.mock;

//import com.firework.cluster.rpc .FireworkQueueServiceGrpc;
//import com.firework.cluster.rpc.FireworkTransfer;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import com.webank.eggroll.core.transfer.TransferServiceGrpc;
import io.grpc.Server;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;

import java.io.IOException;
import java.net.InetSocketAddress;


public class MockServer {


//    private static class  QueueMockService      extends FireworkQueueServiceGrpc.FireworkQueueServiceImplBase{
//
//        public void produceUnary(com.firework.cluster.rpc.FireworkTransfer.ProduceRequest request,
//                                 io.grpc.stub.StreamObserver<com.firework.cluster.rpc.FireworkTransfer.ProduceResponse> responseObserver) {
//            System.err.println("receive produceUnary "+request);
//            FireworkTransfer.ProduceResponse.Builder  builder = FireworkTransfer.ProduceResponse.newBuilder();
//            builder.setCode(0);
//            responseObserver.onNext(builder.build());
//            responseObserver.onCompleted();
//        }
//
//    }

    public static void main(String[] args) {
        InetSocketAddress addr = new InetSocketAddress("localhost", 9375);
        System.err.println("grpc address : " + addr);
        NettyServerBuilder nettyServerBuilder = NettyServerBuilder.forAddress(addr);
        nettyServerBuilder.addService(new MockService());
        nettyServerBuilder.addService(new PtpService());
        nettyServerBuilder.addService(new MockEggpair());
        //nettyServerBuilder.addService(new QueueMockService());
        //nettyServerBuilder.addService(ServerInterceptors.intercept(new QueueMockService(), new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
        Server server = nettyServerBuilder.build();
        try {
            server.start();
            try {
                server.awaitTermination();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.err.println("================");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

//    private static Server buildServer() {
//        NettyServerBuilder nettyServerBuilder = (NettyServerBuilder) ServerBuilder.forPort(9375);
//        nettyServerBuilder.addService(ServerInterceptors.intercept(proxyGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
//        //  nettyServerBuilder.addService(ServerInterceptors.intercept(queueGrpcservice, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
//        //   nettyServerBuilder.addService(ServerInterceptors.intercept(commonService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
//
//        nettyServerBuilder
//                .executor(Executors.newCachedThreadPool())
//                .maxConcurrentCallsPerConnection(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONCURRENT_CALL_PER_CONNECTION)
//                .maxInboundMessageSize(MetaInfo.PROPERTY_GRPC_SERVER_MAX_INBOUND_MESSAGE_SIZE)
//                .maxInboundMetadataSize(MetaInfo.PROPERTY_GRPC_SERVER_MAX_INBOUND_METADATA_SIZE)
//                .flowControlWindow(MetaInfo.PROPERTY_GRPC_SERVER_FLOW_CONTROL_WINDOW);
//        if (MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIME_SEC > 0)
//            nettyServerBuilder.keepAliveTime(MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIMEOUT_SEC > 0)
//            nettyServerBuilder.keepAliveTimeout(MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIMEOUT_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_SERVER_PERMIT_KEEPALIVE_TIME_SEC > 0)
//            nettyServerBuilder.permitKeepAliveTime(MetaInfo.PROPERTY_GRPC_SERVER_PERMIT_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_WITHOUT_CALLS_ENABLED)
//            nettyServerBuilder.permitKeepAliveWithoutCalls(MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_WITHOUT_CALLS_ENABLED);
//        if (MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_IDLE_SEC > 0)
//            nettyServerBuilder.maxConnectionIdle(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_IDLE_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_SEC > 0)
//            nettyServerBuilder.maxConnectionAge(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_GRACE_SEC > 0)
//            nettyServerBuilder.maxConnectionAgeGrace(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_GRACE_SEC, TimeUnit.SECONDS);
//        return nettyServerBuilder.build();
//    }

    private static class PtpService extends PrivateTransferProtocolGrpc.PrivateTransferProtocolImplBase {

        public PtpService() {

        }

        public StreamObserver<Osx.Inbound> transport(
                StreamObserver<Osx.Outbound> responseObserver) {
            return new StreamObserver<Osx.Inbound>() {
                @Override
                public void onNext(Osx.Inbound inbound) {
                    System.err.println(inbound);
                }

                @Override
                public void onError(Throwable throwable) {
                    throwable.printStackTrace();
                }

                @Override
                public void onCompleted() {
                    Proxy.Metadata metadata = Proxy.Metadata.newBuilder().build();
                    Osx.Outbound.Builder outBoundBuilder = Osx.Outbound.newBuilder();
                    outBoundBuilder.setCode(StatusCode.SUCCESS);
                    outBoundBuilder.setMessage(Dict.SUCCESS);
                    outBoundBuilder.setPayload(metadata.toByteString());
                    responseObserver.onNext(outBoundBuilder.build());
                    responseObserver.onCompleted();
                }
            };
        }

        /**
         *
         */
        public void invoke(Osx.Inbound request,
                           StreamObserver<org.ppc.ptp.Osx.Outbound> responseObserver) {
            System.err.println("invoke : " + request);
            Osx.Outbound out = Osx.Outbound.newBuilder().setCode(StatusCode.SUCCESS).setMessage(Dict.SUCCESS).build();
            responseObserver.onNext(out);
            responseObserver.onCompleted();

        }

    }



    private static class MockEggpair extends TransferServiceGrpc.TransferServiceImplBase{

        public io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> send(
                io.grpc.stub.StreamObserver<com.webank.eggroll.core.transfer.Transfer.TransferBatch> responseObserver) {
             return   new StreamObserver<Transfer.TransferBatch>(){
                   @Override
                   public void onNext(Transfer.TransferBatch transferBatch) {
                        System.err.println("======== on next "+ transferBatch);
                   }

                   @Override
                   public void onError(Throwable throwable) {
                       System.err.println("======== on error "+ throwable);
                   }

                   @Override
                   public void onCompleted() {

                       System.err.println("======== on completed ");
                       responseObserver.onCompleted();
                   }
               };


        };

    }
    private static class MockService extends DataTransferServiceGrpc.DataTransferServiceImplBase {

        public void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
                              io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
            System.err.println("receive unaryCall");
            Proxy.Packet.Builder builder = Proxy.Packet.newBuilder();
            builder.setBody(Proxy.Data.newBuilder().setValue(ByteString.copyFromUtf8("my name is god")).build());
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }


        public void pull(com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata request,
                         io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
            System.err.println("receive pull");
            for (int i = 0; i < 10; i++) {
                Proxy.Packet.Builder builder = Proxy.Packet.newBuilder();
                builder.setBody(Proxy.Data.newBuilder().setValue(ByteString.copyFromUtf8("xiaoxiao" + i)).build());
                responseObserver.onNext(builder.build());
                System.err.println("send");
            }
            responseObserver.onCompleted();
        }


        public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> push(
                io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> responseObserver) {
            return new StreamObserver<Proxy.Packet>() {
                @Override
                public void onNext(Proxy.Packet value) {
                    System.err.println("mock receive" + value);
                }

                @Override
                public void onError(Throwable t) {
                    System.err.println("mock onError");
                    t.printStackTrace();
                }

                @Override
                public void onCompleted() {
                    System.err.println("mock receive onCompleted");
                    Proxy.Metadata.Builder builder = Proxy.Metadata.newBuilder();
                    builder.setExt(ByteString.copyFrom("this is hearven".getBytes()));
                    responseObserver.onNext(builder.build());
                    responseObserver.onCompleted();
                }
            };
        }
    }
}
