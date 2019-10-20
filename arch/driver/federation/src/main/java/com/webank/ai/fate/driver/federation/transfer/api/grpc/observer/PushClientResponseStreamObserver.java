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

package com.webank.ai.fate.driver.federation.transfer.api.grpc.observer;

import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.eggroll.core.api.grpc.observer.BaseCallerResponseStreamObserver;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class PushClientResponseStreamObserver extends BaseCallerResponseStreamObserver<Proxy.Packet, Proxy.Metadata> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ToStringUtils toStringUtils;
    private TransferBroker transferBroker;
    // private DelayedResult<Metadata> delayedResult;
    private Proxy.Metadata metadata;

    public PushClientResponseStreamObserver(CountDownLatch finishLatch, TransferBroker transferBroker) {
        super(finishLatch);
        this.transferBroker = transferBroker;
    }

    @Override
    public void onNext(Proxy.Metadata metadata) {
        LOGGER.info("[PUSH][CLIENTOBSERVER][ONNEXT]SendClientResponseStreamObserver.onNext. metadata: {}", toStringUtils.toOneLineString(metadata));
        this.metadata = metadata;
        // this.delayedResult.setResult(metadata);
    }

    @Override
    public void onError(Throwable throwable) {
        LOGGER.info("[PUSH][CLIENTOBSERVER][ONERROR]SendClientResponseStreamObserver.onError. metadata: {}", toStringUtils.toOneLineString(metadata));
        super.onError(throwable);
    }

    @Override
    public void onCompleted() {
        LOGGER.info("[PUSH][CLIENTOBSERVER][ONCOMPLETE]SendClientResponseStreamObserver.onComplete. metadata: {}", toStringUtils.toOneLineString(metadata));
        super.onCompleted();
    }
}
