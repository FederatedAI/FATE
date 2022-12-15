
package com.osx.broker.grpc;

//import com.firework.transfer.service.PullService;

import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.OutboundPackage;
import com.osx.broker.service.PushService2;
import com.osx.broker.service.UnaryCallService;
import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ProxyGrpcService extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    Logger logger = LoggerFactory.getLogger(ProxyGrpcService.class);

    public ProxyGrpcService(PushService2 pushService2,
                            UnaryCallService unaryCallService
                     ){
        this.pushService2 = pushService2;
        this.unaryCallService = unaryCallService;


    }

    UnaryCallService unaryCallService;
    PushService2 pushService2;
    /**
     */

    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> push(
            io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata> responseObserver) {
        try {
            logger.info("receive push request");
            Context context = ContextUtil.buildContext();
            InboundPackage<PushRequestDataWrap> data = new InboundPackage<>();
            PushRequestDataWrap pushRequestDataWrap = new PushRequestDataWrap();
            pushRequestDataWrap.setStreamObserver(responseObserver);
            data.setBody(pushRequestDataWrap);
            OutboundPackage<StreamObserver> outboundPackage = pushService2.service(context, data);
            return outboundPackage.getData();
        }catch(Exception e){
            e.printStackTrace();
        }
        return null;
    }


//    /**
//     */
//    @RegisterService(serviceName = "pull")
//    public void pull(com.webank.ai.eggroll.api.networking.proxy.Proxy.Metadata request,
//                     io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
//        Context  context = buildContext();
//        InboundPackage<PullRequestDataWrap> data = new InboundPackage<>();
//        PullRequestDataWrap pullRequestDataWrap = new  PullRequestDataWrap();
//        pullRequestDataWrap.setMetadata(request);
//        pullRequestDataWrap.setStreamObserver(responseObserver);
//        data.setBody(pullRequestDataWrap);
//        OutboundPackage<StreamObserver<Proxy.Packet>>  outboundPackage = pullService.service(context,data);
//    }

    /**
     */

    public void unaryCall(com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet request,
                          io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.Packet> responseObserver) {
        Context context = ContextUtil.buildContext();
        InboundPackage<Proxy.Packet> data = new InboundPackage<>();
        data.setBody(request);
        context.setDataSize(request.getSerializedSize());
        OutboundPackage<Proxy.Packet> outboundPackage =  unaryCallService.service(context,data);
        Proxy.Packet result = outboundPackage.getData();
        Throwable  throwable = outboundPackage.getThrowable();
        if(throwable!=null) {
            responseObserver.onError(throwable);
        }else {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }
    }


    public io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> polling(
            io.grpc.stub.StreamObserver<com.webank.ai.eggroll.api.networking.proxy.Proxy.PollingFrame> responseObserver) {
        return null;
    }



}
