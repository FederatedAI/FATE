package com.osx.broker.mock;

//import com.firework.cluster.rpc .FireworkQueueServiceGrpc;
//import com.firework.cluster.rpc.FireworkTransfer;

import com.osx.core.constant.Dict;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.StatusCode;
import com.osx.broker.grpc.ContextPrepareInterceptor;
import com.osx.broker.grpc.ServiceExceptionHandler;
import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Pcp;
import org.ppc.ptp.PrivateTransferProtocolGrpc;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static com.osx.broker.ServiceContainer.*;

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

    private static class PtpService  extends PrivateTransferProtocolGrpc.PrivateTransferProtocolImplBase{

        public PtpService(){

        }

        public StreamObserver<org.ppc.ptp.Pcp.Inbound> transport(
                StreamObserver<org.ppc.ptp.Pcp.Outbound> responseObserver) {
            return  new  StreamObserver<Pcp.Inbound>(){
                @Override
                public void onNext(Pcp.Inbound inbound) {
                    System.err.println(inbound);
                }
                @Override
                public void onError(Throwable throwable) {
                        throwable.printStackTrace();
                }
                @Override
                public void onCompleted() {
                    Proxy.Metadata  metadata= Proxy.Metadata.newBuilder().build();
                    Pcp.Outbound.Builder  outBoundBuilder =  Pcp.Outbound.newBuilder();
                    outBoundBuilder.setCode(StatusCode.SUCCESS);
                    outBoundBuilder.setMessage(Dict.SUCCESS);
                    outBoundBuilder.setPayload(metadata.toByteString());
                    responseObserver.onNext(outBoundBuilder.build());
                    responseObserver.onCompleted();
                }
            };
        }

        /**
         */
        public void invoke(Pcp.Inbound request,
                           StreamObserver<org.ppc.ptp.Pcp.Outbound> responseObserver) {
                System.err.println("invoke : "+request);
            Pcp.Outbound out = org.ppc.ptp.Pcp.Outbound.newBuilder().setCode(StatusCode.SUCCESS).setMessage(Dict.SUCCESS).build();
            responseObserver.onNext(out);
            responseObserver.onCompleted();

        }

    }


    private static class  MockService extends DataTransferServiceGrpc.DataTransferServiceImplBase{

        public void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
                              io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
            System.err.println("receive unaryCall");
            Proxy.Packet.Builder  builder = Proxy.Packet.newBuilder();
            builder.setBody(Proxy.Data.newBuilder().setValue(ByteString.copyFromUtf8("my name is god")).build());
            responseObserver.onNext(builder.build());
            responseObserver.onCompleted();
        }


        public void pull(com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata request,
                         io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
           System.err.println("receive pull");
           for(int i=0;i<10;i++){
               Proxy.Packet.Builder  builder = Proxy.Packet.newBuilder();
               builder.setBody(Proxy.Data.newBuilder().setValue(ByteString.copyFromUtf8("xiaoxiao"+i)).build());
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
                    System.err.println("mock receive"+ value);
                }

                @Override
                public void onError(Throwable t) {
                    System.err.println("mock onError");
                    t.printStackTrace();
                }

                @Override
                public void onCompleted() {
                    System.err.println("mock receive onCompleted");
                    Proxy.Metadata.Builder builder =  Proxy.Metadata.newBuilder();
                    builder.setExt(ByteString.copyFrom("this is hearven".getBytes()));
                    responseObserver.onNext(builder.build());
                    responseObserver.onCompleted();
                }
            };
        }
    }

    public  static  void main(String[]  args){
        InetSocketAddress addr = new InetSocketAddress("localhost", 9375);
        System.err.println("grpc address : "+addr);
        NettyServerBuilder nettyServerBuilder = NettyServerBuilder.forAddress(addr);
        nettyServerBuilder.addService(new  MockService());
        nettyServerBuilder.addService(new  PtpService());
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


    private static Server buildServer( ){
        NettyServerBuilder nettyServerBuilder = (NettyServerBuilder) ServerBuilder.forPort(9375);
        nettyServerBuilder.addService(ServerInterceptors.intercept(proxyGrpcService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
      //  nettyServerBuilder.addService(ServerInterceptors.intercept(queueGrpcservice, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
     //   nettyServerBuilder.addService(ServerInterceptors.intercept(commonService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));

        nettyServerBuilder
                .executor(Executors.newCachedThreadPool())
                .maxConcurrentCallsPerConnection(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONCURRENT_CALL_PER_CONNECTION)
                .maxInboundMessageSize(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_MESSAGE_SIZE)
                .maxInboundMetadataSize(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_METADATA_SIZE)
                .flowControlWindow(MetaInfo.PROPERTY_GRPC_CHANNEL_FLOW_CONTROL_WINDOW);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC > 0) nettyServerBuilder.keepAliveTime(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC > 0) nettyServerBuilder.keepAliveTimeout(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC > 0) nettyServerBuilder.permitKeepAliveTime(MetaInfo.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED) nettyServerBuilder.permitKeepAliveWithoutCalls(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC > 0) nettyServerBuilder.maxConnectionIdle(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC > 0) nettyServerBuilder.maxConnectionAge(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC > 0) nettyServerBuilder.maxConnectionAgeGrace(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC, TimeUnit.SECONDS);
        return  nettyServerBuilder.build();
    }
}
