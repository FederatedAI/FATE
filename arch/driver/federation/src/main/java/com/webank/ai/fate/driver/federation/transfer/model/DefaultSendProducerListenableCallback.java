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

package com.webank.ai.fate.driver.federation.transfer.model;

import com.webank.ai.eggroll.api.core.BasicMeta;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.List;

@Component
@Scope("prototype")
public class DefaultSendProducerListenableCallback implements ListenableFutureCallback<BasicMeta.ReturnStatus> {
    private static final Logger LOGGER = LogManager.getLogger();
    private final List<BasicMeta.ReturnStatus> results;
    private final TransferBroker transferBroker;
    private final List<Throwable> errorContainers;
    private final String ip;
    private final int port;

    public DefaultSendProducerListenableCallback(List<BasicMeta.ReturnStatus> results,
                                                 TransferBroker transferBroker,
                                                 List<Throwable> errorContainers,
                                                 String ip,
                                                 int port) {
        this.results = results;
        this.transferBroker = transferBroker;
        this.errorContainers = errorContainers;
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void onFailure(Throwable ex) {
        LOGGER.info("[FEDERATION][PRODUCER][CALLBACK] onFailure: {}:{}", ip, port);
        errorContainers.add(ex);
        transferBroker.setError(ex);
    }

    @Override
    public void onSuccess(BasicMeta.ReturnStatus result) {
        LOGGER.info("[FEDERATION][PRODUCER][CALLBACK] onSuccess: {}:{}", ip, port);
        results.add(result);
        transferBroker.setFinished();
    }
}
