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

import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.manager.TransferMetaHelper;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;

public abstract class BaseRecvConsumeAction implements TransferQueueConsumeAction {
    private final Logger LOGGER = LogManager.getLogger();
    @Autowired
    protected TypeConversionUtils typeConversionUtils;
    @Autowired
    protected ErrorUtils errorUtils;
    @Autowired
    protected TransferMetaHelper transferMetaHelper;
    @Autowired
    protected TransferPojoUtils transferPojoUtils;
    @Autowired
    protected ToStringUtils toStringUtils;
    @Autowired
    protected RecvBrokerManager recvBrokerManager;

    protected TransferBroker transferBroker;
    protected Federation.TransferMeta transferMeta;
    protected String transferMetaId;
    private Federation.TransferStatus currentTransferStatus;

    public BaseRecvConsumeAction(TransferBroker transferBroker) {
        this.transferBroker = transferBroker;
        this.transferMeta = transferBroker.getTransferMeta();
    }

    @Override
    public void onInit() {
        transferMetaId = transferPojoUtils.generateTransferId(transferMeta);
        // transferMetaHelper.onInit(transferMeta);
        currentTransferStatus = Federation.TransferStatus.INITIALIZING;
    }

    @Override
    public void onProcess() {
        if (currentTransferStatus != Federation.TransferStatus.PROCESSING) {
            // transferMetaHelper.onProcess(transferMeta);
            currentTransferStatus = Federation.TransferStatus.PROCESSING;
        }
    }

    @Override
    public void onComplete() {
        if (!transferBroker.hasError()) {
            // transferMetaHelper.onComplete(transferMeta);
            currentTransferStatus = Federation.TransferStatus.COMPLETE;
        }
        LOGGER.info("[FEDERATION][RECV] onComplete transferMeta: {}",
                toStringUtils.toOneLineString(transferMeta));
    }
}
