package com.osx.broker.ptp;

import com.osx.broker.callback.CompleteCallback;
import com.osx.broker.callback.ErrorCallback;
import com.osx.broker.util.TransferUtil;
import com.osx.core.context.Context;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PtpForwardPushRespSO implements StreamObserver<Osx.Outbound> {

    Logger logger = LoggerFactory.getLogger(PtpForwardPushRespSO.class);
    String sourceType;
    StreamObserver backPushRespSO;
    CompleteCallback completeCallback;
    ErrorCallback errorCallback;
    Context context;

    public PtpForwardPushRespSO(Context context, StreamObserver backPushRespSO, String sourceType, CompleteCallback completeCallback, ErrorCallback errorCallback) {

        this.sourceType = sourceType;
        this.backPushRespSO = backPushRespSO;
        this.context = context;
        this.completeCallback = completeCallback;
        this.errorCallback = errorCallback;
    }

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
    public void onNext(Osx.Outbound value) {

        if (sourceType.equals("proxy")) {
            Proxy.Metadata metadata = TransferUtil.buildProxyMetadataFromOutbound(value);
            backPushRespSO.onNext(metadata);
        } else {
            backPushRespSO.onNext(value);
        }
    }

    @Override
    public void onError(Throwable t) {
        logger.error("onError {}", t);
        if (errorCallback != null) {
            errorCallback.callback(t);
        }
        backPushRespSO.onError(t);

    }

    @Override
    public void onCompleted() {
        if (completeCallback != null) {
            completeCallback.callback();
        }
        backPushRespSO.onCompleted();
    }
}
