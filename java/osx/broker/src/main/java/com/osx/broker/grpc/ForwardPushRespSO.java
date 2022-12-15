package com.osx.broker.grpc;

import com.osx.core.context.Context;
import com.osx.broker.callback.CompleteCallback;
import com.osx.broker.callback.ErrorCallback;
//import com.firework.transfer.service.TokenApplyService;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ForwardPushRespSO implements StreamObserver<Proxy.Metadata> {

    Logger logger = LoggerFactory.getLogger(ForwardPushRespSO.class);

    public  ForwardPushRespSO(Context context, StreamObserver  backPushRespSO , CompleteCallback completeCallback, ErrorCallback errorCallback){
        this.backPushRespSO = backPushRespSO;
        this.context = context;
        this.completeCallback = completeCallback;
        this.errorCallback = errorCallback;
    }

    StreamObserver  backPushRespSO;

    CompleteCallback  completeCallback;

    ErrorCallback  errorCallback;

    Context  context;

    public StreamObserver getBackPushRespSO() {
        return backPushRespSO;
    }

    public void setBackPushRespSO(StreamObserver backPushRespSO) {
        this.backPushRespSO = backPushRespSO;
    }

//    public TokenApplyService getTokenApplyService() {
//        return tokenApplyService;
//    }
//
//    public void setTokenApplyService(TokenApplyService tokenApplyService) {
//        this.tokenApplyService = tokenApplyService;
//    }

    //TokenApplyService   tokenApplyService;
    @Override
    public void onNext(Proxy.Metadata value) {
        backPushRespSO.onNext(value);
    }

    @Override
    public void onError(Throwable t) {
        logger.error("onError {}",t);
        if(errorCallback!=null) {
            errorCallback.callback(t);
        }
        backPushRespSO.onError(t);

    }

    @Override
    public void onCompleted() {
        if(completeCallback!=null){
            completeCallback.callback();
        }
        backPushRespSO.onCompleted();
    }
}
