package com.osx.broker.grpc;

import com.firework.cluster.rpc.Firework;
import com.firework.cluster.rpc.FireworkServiceGrpc;
import com.osx.broker.ServiceContainer;
import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.OutboundPackage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ClusterService extends FireworkServiceGrpc.FireworkServiceImplBase {

    Logger logger = LoggerFactory.getLogger(ClusterService.class);

    //ZookeeperRegistry zookeeperRegistry;

//    RouterTableService routerTableService;
//
//    DataTransferService dataTransferService;


    //TransferQueueApplyService    transferQueueApplyService;


//    public TokenService getTokenService() {
//        return tokenService;
//    }
//
//
//    DefaultTokenService tokenService;
//
//    public HeartbeatService getHeartbeatService() {
//        return heartbeatService;
//    }
//
//    public void setHeartbeatService(HeartbeatService heartbeatService) {
//        this.heartbeatService = heartbeatService;
//    }

//    HeartbeatService  heartbeatService;
//
//    QueryFlowRuleService queryFlowRuleService;

//    public void heartBeat(com.firework.cluster.rpc.Firework.HeartbeatRequest request,
//                          io.grpc.stub.StreamObserver<com.firework.cluster.rpc.Firework.HeartbeatResponse> responseObserver) {
//        InboundPackage inboundPackage = new  InboundPackage();
//        inboundPackage.setBody(request);
//        Context context = buidlContext();
//        OutboundPackage<Firework.HeartbeatResponse>  outboundPackage  = heartbeatService.service(context,inboundPackage);
//        responseObserver.onNext(Firework.HeartbeatResponse.newBuilder().build());
//        responseObserver.onCompleted();
//    }


    public void applyToken(com.firework.cluster.rpc.Firework.TokenRequest request,
                           io.grpc.stub.StreamObserver<com.firework.cluster.rpc.Firework.TokenResponse> responseObserver) {
        Context context = buidlContext();
        InboundPackage inboundPackage = new InboundPackage();
        inboundPackage.setBody(request);
        OutboundPackage<Firework.TokenResponse> outboundPackage = ServiceContainer.defaultTokenService.service(context, inboundPackage);
        Firework.TokenResponse tokenResponse = outboundPackage.getData();
        responseObserver.onNext(tokenResponse);
        responseObserver.onCompleted();
    }


    public void queryRouter(com.firework.cluster.rpc.Firework.QueryRouterRequest request,
                            io.grpc.stub.StreamObserver<com.firework.cluster.rpc.Firework.TokenResponse> responseObserver) {

    }

//    public void queryFlowRule(com.firework.cluster.rpc.Firework.QueryFlowRuleRequest request,
//                              io.grpc.stub.StreamObserver<com.firework.cluster.rpc.Firework.QueryFlowRuleResponse> responseObserver) {
//        Context context = buidlContext();
//        InboundPackage inboundPackage = new  InboundPackage();
//        inboundPackage.setBody(request);
//        OutboundPackage<Firework.QueryFlowRuleResponse>  outboundPackage =  queryFlowRuleService.service(context,inboundPackage);
//        Firework.QueryFlowRuleResponse queryFlowRuleResponse = outboundPackage.getData();
//        responseObserver.onNext(queryFlowRuleResponse);
//        responseObserver.onCompleted();
//    }

//    public void applyTransferQueue(com.firework.cluster.rpc.Firework.ApplyTransferQueueRequest request,
//                                   io.grpc.stub.StreamObserver<com.firework.cluster.rpc.Firework.ApplyTransferQueueResponse> responseObserver) {
//        Context context  = buidlContext();
//        InboundPackage inboundPackage = new  InboundPackage();
//        inboundPackage.setBody(request);
//        OutboundPackage<Firework.ApplyTransferQueueResponse>  outboundPackage =  ServiceContainer.clusterQueueApplyService.service(context,inboundPackage);
//        responseObserver.onNext(outboundPackage.getData());
//        responseObserver.onCompleted();
//    }


//    public void unRegisterTransferQueue(com.firework.cluster.rpc.Firework.UnRegisterTransferQueueRequest request,
//                                        io.grpc.stub.StreamObserver<com.firework.cluster.rpc.Firework.UnRegisterTransferQueueResponse> responseObserver) {
//
//        Firework.UnRegisterTransferQueueResponse response = ServiceContainer.transferQueueApplyManager.unRegisterTransferQueue(request);
//        responseObserver.onNext(response);
//        responseObserver.onCompleted();
//    }

    /**
     *
     */
    public void cancelClusterTransfer(com.firework.cluster.rpc.Firework.CancelClusterTransferRequest request,
                                      io.grpc.stub.StreamObserver<com.firework.cluster.rpc.Firework.CancelClusterTransferResponse> responseObserver) {

    }


    private Context buidlContext() {
        Context context = new Context();
        // context.setSourceIp(sourceIp.get()!=null?sourceIp.get().toString():"");
        return context;
    }

//    private Server buildServer( ){
//        NettyServerBuilder nettyServerBuilder = (NettyServerBuilder) ServerBuilder.forPort(MetaInfo.PROPERTY_PORT);
//        nettyServerBuilder.addService(ServerInterceptors.intercept(this, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
//        //ettyServerBuilder.addService(ServerInterceptors.intercept(queueGrpcservice, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
//        nettyServerBuilder
//                .executor(Executors.newCachedThreadPool())
//                .maxConcurrentCallsPerConnection(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONCURRENT_CALL_PER_CONNECTION)
//                .maxInboundMessageSize(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_MESSAGE_SIZE)
//                .maxInboundMetadataSize(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_METADATA_SIZE)
//                .flowControlWindow(MetaInfo.PROPERTY_GRPC_CHANNEL_FLOW_CONTROL_WINDOW);
//        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC > 0) nettyServerBuilder.keepAliveTime(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC > 0) nettyServerBuilder.keepAliveTimeout(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC > 0) nettyServerBuilder.permitKeepAliveTime(MetaInfo.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED) nettyServerBuilder.permitKeepAliveWithoutCalls(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED);
//        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC > 0) nettyServerBuilder.maxConnectionIdle(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC > 0) nettyServerBuilder.maxConnectionAge(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC, TimeUnit.SECONDS);
//        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC > 0) nettyServerBuilder.maxConnectionAgeGrace(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC, TimeUnit.SECONDS);
//        return  nettyServerBuilder.build();
//    }

}
