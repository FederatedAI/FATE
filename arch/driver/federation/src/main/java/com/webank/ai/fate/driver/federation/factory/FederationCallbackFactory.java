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

package com.webank.ai.fate.driver.federation.factory;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.fate.driver.federation.transfer.model.DefaultConsumerListenableCallback;
import com.webank.ai.fate.driver.federation.transfer.model.DefaultSendProducerListenableCallback;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.CountDownLatch;

@Component
public class FederationCallbackFactory {
    @Autowired
    private ApplicationContext applicationContext;

    public DefaultConsumerListenableCallback createDefaultConsumerListenableCallback(final List<Throwable> errorContainer,
                                                                                     final CountDownLatch finishLatch,
                                                                                     final String ip,
                                                                                     final int port) {
        return applicationContext.getBean(
                DefaultConsumerListenableCallback.class,
                errorContainer,
                finishLatch,
                ip,
                port);
    }

    public DefaultSendProducerListenableCallback createDtableSendProducerListenableCallback(final List<BasicMeta.ReturnStatus> results,
                                                                                            final TransferBroker transferBroker,
                                                                                            final List<Throwable> errorContainer,
                                                                                            final String ip,
                                                                                            final int port) {
        return applicationContext.getBean(
                DefaultSendProducerListenableCallback.class,
                results,
                transferBroker,
                errorContainer,
                ip,
                port);
    }
}
