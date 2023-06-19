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
package com.osx.broker.grpc;

import com.osx.api.context.Context;
import com.osx.broker.callback.CompleteCallback;
import com.osx.broker.callback.ErrorCallback;
import com.osx.broker.util.TransferUtil;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ForwardPushRespSO implements StreamObserver<Proxy.Metadata> {

    Logger logger = LoggerFactory.getLogger(ForwardPushRespSO.class);
    StreamObserver backPushRespSO;
    Class  backPushRespClass;
    CompleteCallback completeCallback;
    ErrorCallback errorCallback;
    Context context;


    public ForwardPushRespSO(Context context, StreamObserver backPushRespSO,    Class  backPushRespClass, CompleteCallback completeCallback, ErrorCallback errorCallback) {
        this.backPushRespSO = backPushRespSO;
        this.backPushRespClass = backPushRespClass;
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
        if(backPushRespClass.equals(Proxy.Metadata.class)) {
            backPushRespSO.onNext(value);
        }else{
            Osx.Outbound  outbound = TransferUtil.buildOutboundFromProxyMetadata(value);
            backPushRespSO.onNext(outbound);
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


    public  static  void main(String[] args){
        System.err.println(Transfer.TransferBatch.class.getCanonicalName());
    }
}
