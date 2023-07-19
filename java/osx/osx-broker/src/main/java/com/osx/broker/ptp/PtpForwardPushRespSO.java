/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.osx.broker.ptp;

import com.osx.api.context.Context;
import com.osx.broker.callback.CompleteCallback;
import com.osx.broker.callback.ErrorCallback;
import com.osx.broker.util.TransferUtil;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PtpForwardPushRespSO implements StreamObserver<Osx.Outbound> {

    Logger logger = LoggerFactory.getLogger(PtpForwardPushRespSO.class);

    StreamObserver backPushRespSO;
    Class backPushRespSOClass;
    CompleteCallback completeCallback;
    ErrorCallback errorCallback;
    Context context;

    public PtpForwardPushRespSO(Context context, StreamObserver backPushRespSO, Class backPushRespSOClass , CompleteCallback completeCallback, ErrorCallback errorCallback) {

        this.backPushRespSOClass = backPushRespSOClass;
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

        if (backPushRespSOClass.equals(Proxy.Metadata.class)) {
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
