//package com.osx.broker.grpc;
//
//import com.osx.broker.constants.Direction;
//import com.osx.broker.router.FateRouterService;
//import com.osx.broker.util.ResourceUtil;
//import com.osx.core.config.MetaInfo;
//import com.osx.core.context.Context;
//import com.osx.core.exceptions.NoRouterInfoException;
//import com.osx.core.frame.GrpcConnectionFactory;
//import com.osx.core.router.RouterInfo;
//import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
//import com.webank.ai.eggroll.api.networking.proxy.Proxy;
//import io.grpc.ManagedChannel;
//import io.grpc.stub.StreamObserver;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.util.concurrent.CountDownLatch;
//import java.util.concurrent.TimeUnit;
//import java.util.concurrent.TimeoutException;
//
//public class PushReqStreamObserver implements StreamObserver<Proxy.Packet> {
//
//    Logger logger = LoggerFactory.getLogger(PushReqStreamObserver.class);
//
//    boolean init = false;
//
//    Context context;
//
//    CountDownLatch finishLatch;
//
////    public TokenApplyService getTokenApplyService() {
////        return tokenApplyService;
////    }
////
////    public void setTokenApplyService(TokenApplyService tokenApplyService) {
////        this.tokenApplyService = tokenApplyService;
////    }
////
////    TokenApplyService  tokenApplyService;
//    FateRouterService fateRouterService;
//    private StreamObserver<Proxy.Packet> forwardPushReqSO;
//    private StreamObserver<Proxy.Metadata> backRespSO;
//
//
//    public PushReqStreamObserver(Context context, StreamObserver backRespSO, CountDownLatch finishLatch,
//                                 FateRouterService fateRouterService
//                                 //, TokenApplyService tokenApplyService
//    ) {
//        this.backRespSO = backRespSO;
//        this.context = context.subContext();
//        this.context.setServiceName("pushTransfer");
//        this.finishLatch = finishLatch;
//        this.fateRouterService = fateRouterService;
//        // this.tokenApplyService = tokenApplyService;
//    }
//
//    public FateRouterService getFateRouterService() {
//        return fateRouterService;
//    }
//
//    public void setFateRouterService(FateRouterService fateRouterService) {
//        this.fateRouterService = fateRouterService;
//    }
//
//    private void init(Proxy.Packet value) {
//        RouterInfo routerInfo = fateRouterService.route(value);
//        if (routerInfo != null) {
//            context.setRouterInfo(routerInfo);
//        } else {
//            throw new NoRouterInfoException("no router");
//        }
//        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(context.getRouterInfo());
//        DataTransferServiceGrpc.DataTransferServiceStub stub = DataTransferServiceGrpc.newStub(managedChannel);
//        ForwardPushRespSO forwardPushRespSO = new ForwardPushRespSO(context, backRespSO, () -> {
//            finishLatch.countDown();
//        }, (t) -> {
//            finishLatch.countDown();
//        });
//        //  forwardPushRespSO.setTokenApplyService(tokenApplyService);
//        forwardPushReqSO = stub.push(forwardPushRespSO);
//        init = true;
//    }
//
//
//    @Override
//    public void onNext(Proxy.Packet value) {
//        if (!init) {
//            init(value);
//        }
//        byte[] data = value.toByteArray();
//        int size = data.length;
//        String resource = ResourceUtil.buildResource(context.getRouterInfo(), Direction.UP);
//        // tokenApplyService.applyToken(context,resource,size);
//        forwardPushReqSO.onNext(value);
//    }
//
//
//    @Override
//    public void onError(Throwable t) {
//        logger.info("onError");
//        if (forwardPushReqSO != null) {
//            forwardPushReqSO.onError(t);
//        }
//        context.printFlowLog();
//    }
//
//    @Override
//    public void onCompleted() {
//
//        boolean needPrintFlow = true;
//        if (forwardPushReqSO != null) {
//            forwardPushReqSO.onCompleted();
//            try {
//                if (!finishLatch.await(MetaInfo.PROPERTY_GRPC_ONCOMPLETED_WAIT_TIMEOUT, TimeUnit.SECONDS)) {
//                    onError(new TimeoutException());
//                    needPrintFlow = false;
//                }
//            } catch (InterruptedException e) {
//                onError(e);
//                needPrintFlow = false;
//            }
//        }
//        if (needPrintFlow) {
//            context.printFlowLog();
//        }
//
//        logger.info("receive completed  !!!!");
//    }
//}
