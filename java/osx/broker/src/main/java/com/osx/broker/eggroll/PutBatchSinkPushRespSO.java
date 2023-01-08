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
package com.osx.broker.eggroll;
import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.core.exceptions.ParameterException;
import com.osx.core.utils.ToStringUtils;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;

public class PutBatchSinkPushRespSO implements StreamObserver<Transfer.TransferBatch> {

    StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO;
    Proxy.Metadata reqHeader;
    Future<ErTask> commandFuture;
    CountDownLatch finishLatch;
    Logger logger = LoggerFactory.getLogger(PutBatchSinkPushRespSO.class);

    public PutBatchSinkPushRespSO(Proxy.Metadata reqHeader,
                                  Future<ErTask> commandFuture,
                                  StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO,
                                  CountDownLatch finishLatch
    ) {
        this.reqHeader = reqHeader;
        this.commandFuture = commandFuture;
        this.eggSiteServicerPushRespSO = eggSiteServicerPushRespSO;
        this.finishLatch = finishLatch;
    }

    @Override
    public void onNext(Transfer.TransferBatch resp) {
        try {
            commandFuture.get();
            eggSiteServicerPushRespSO.onNext(reqHeader.toBuilder().setAck(resp.getHeader().getId()).build());
            eggSiteServicerPushRespSO.onCompleted();
        } catch (Exception e) {
          logger.error("send to eggroll error",e);
        }
    }

    @Override
    public void onError(Throwable throwable) {
        eggSiteServicerPushRespSO.onError(throwable);
    }

    @Override
    public void onCompleted() {
        finishLatch.countDown();
        eggSiteServicerPushRespSO.onCompleted();
    }
}


