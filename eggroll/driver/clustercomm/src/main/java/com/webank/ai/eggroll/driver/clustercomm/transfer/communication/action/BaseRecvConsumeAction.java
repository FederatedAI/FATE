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

package com.webank.ai.eggroll.driver.clustercomm.transfer.communication.action;

import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.eggroll.driver.clustercomm.transfer.manager.RecvBrokerManager;
import com.webank.ai.eggroll.driver.clustercomm.transfer.manager.TransferMetaHelper;
import com.webank.ai.eggroll.driver.clustercomm.transfer.utils.TransferPojoUtils;
import com.webank.ai.eggroll.driver.clustercomm.transfer.model.TransferBroker;
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
    protected ClusterComm.TransferMeta transferMeta;
    protected String transferMetaId;
    private ClusterComm.TransferStatus currentTransferStatus;

    public BaseRecvConsumeAction(TransferBroker transferBroker) {
        this.transferBroker = transferBroker;
        this.transferMeta = transferBroker.getTransferMeta();
    }

    @Override
    public void onInit() {
        transferMetaId = transferPojoUtils.generateTransferId(transferMeta);
        // transferMetaHelper.onInit(transferMeta);
        currentTransferStatus = ClusterComm.TransferStatus.INITIALIZING;
    }

    @Override
    public void onProcess() {
        if (currentTransferStatus != ClusterComm.TransferStatus.PROCESSING) {
            // transferMetaHelper.onProcess(transferMeta);
            currentTransferStatus = ClusterComm.TransferStatus.PROCESSING;
        }
    }

    @Override
    public void onComplete() {
        if (!transferBroker.hasError()) {
            // transferMetaHelper.onComplete(transferMeta);
            currentTransferStatus = ClusterComm.TransferStatus.COMPLETE;
        }
        LOGGER.info("[CLUSTERCOMM][RECV] onComplete transferMeta: {}",
                toStringUtils.toOneLineString(transferMeta));
    }
}
