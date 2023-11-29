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
package org.fedai.osx.broker.grpc;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.broker.callback.CompleteCallback;
import org.fedai.osx.broker.callback.ErrorCallback;
import org.fedai.osx.core.context.OsxContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ForwardPushRespSO implements StreamObserver<Proxy.Metadata> {
    Logger logger = LoggerFactory.getLogger(ForwardPushRespSO.class);
    StreamObserver backPushRespSO;
    CompleteCallback completeCallback;
    ErrorCallback errorCallback;
    OsxContext context;

    public ForwardPushRespSO(OsxContext context, StreamObserver backPushRespSO, CompleteCallback completeCallback, ErrorCallback errorCallback) {
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

    @Override
    public void onNext(Proxy.Metadata value) {
        //     if(backPushRespClass.equals(Proxy.Metadata.class)) {
        backPushRespSO.onNext(value);
//        }else{
//            Osx.Outbound  outbound = TransferUtil.buildOutboundFromProxyMetadata(value);
//            backPushRespSO.onNext(outbound);
//        }
    }

    @Override
    public void onError(Throwable t) {
        logger.error("stream stream {} to {} return error",context.getTopic(), context.getRouterInfo(), t);
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
