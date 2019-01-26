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

package com.webank.ai.fate.driver.federation.transfer.communication.action;

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.fate.driver.federation.transfer.api.grpc.client.ProxyClient;
import com.webank.ai.fate.driver.federation.transfer.manager.TransferMetaHelper;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;


@Component
@Scope("prototype")
public class SendConsumeAction implements TransferQueueConsumeAction {
    @Autowired
    protected TransferMetaHelper transferMetaHelper;
    @Autowired
    private ProxyClient proxyClient;
    private Federation.TransferMeta transferMeta;
    private Federation.TransferStatus currentTransferStatus;

    private TransferBroker transferBroker;
    private BasicMeta.Endpoint target;

    public SendConsumeAction(TransferBroker transferBroker, BasicMeta.Endpoint target) {
        this.transferBroker = transferBroker;
        this.target = target;
        this.transferMeta = transferBroker.getTransferMeta();
    }

    @Override
    public void onInit() {
        transferMetaHelper.onInit(transferMeta);
        currentTransferStatus = Federation.TransferStatus.INITIALIZING;

        proxyClient.initPush(transferBroker, target);
    }

    @Override
    public void onProcess() {
        if (currentTransferStatus != Federation.TransferStatus.PROCESSING) {
            transferMetaHelper.onProcess(transferMeta);
            currentTransferStatus = Federation.TransferStatus.PROCESSING;
        }
        proxyClient.doPush();
    }

    @Override
    public void onComplete() {
        proxyClient.completePush();
        if (!transferBroker.hasError()) {
            transferMetaHelper.onComplete(transferMeta);
            currentTransferStatus = Federation.TransferStatus.COMPLETE;
        }
    }
}
