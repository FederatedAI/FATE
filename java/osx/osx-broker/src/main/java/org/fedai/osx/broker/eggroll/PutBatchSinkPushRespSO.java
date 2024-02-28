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
package org.fedai.osx.broker.eggroll;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.router.RouterInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class PutBatchSinkPushRespSO implements StreamObserver<Transfer.TransferBatch> {

    StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO;
    Proxy.Metadata reqHeader;
    Future<ErTask> commandFuture;
    CountDownLatch finishLatch;
    RouterInfo routerInfo;
    Logger logger = LoggerFactory.getLogger(PutBatchSinkPushRespSO.class);

    public PutBatchSinkPushRespSO(Proxy.Metadata reqHeader,
                                  Future<ErTask> commandFuture,
                                  StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO,
                                  CountDownLatch finishLatch, RouterInfo routerInfo
                                  ) {
        this.routerInfo = routerInfo;
        this.reqHeader = reqHeader;
        this.commandFuture = commandFuture;
        this.eggSiteServicerPushRespSO = eggSiteServicerPushRespSO;
        this.finishLatch = finishLatch;
    }

    @Override
    public void onNext(Transfer.TransferBatch resp) {
        try {
            commandFuture.get(MetaInfo.BATCH_SINK_PUSH_EXECUTOR_TIMEOUT, TimeUnit.MILLISECONDS);
            eggSiteServicerPushRespSO.onNext(reqHeader.toBuilder().setAck(resp.getHeader().getId()).build());
//            eggSiteServicerPushRespSO.onCompleted();
        } catch (Exception e) {
            logger.error("send to eggroll error", e);
        }
    }

    @Override
    public void onError(Throwable throwable) {
        logger.error("eggpair {} return error",routerInfo,throwable);
        eggSiteServicerPushRespSO.onError(throwable);
    }

    @Override
    public void onCompleted() {
        finishLatch.countDown();
        eggSiteServicerPushRespSO.onCompleted();
    }
}


