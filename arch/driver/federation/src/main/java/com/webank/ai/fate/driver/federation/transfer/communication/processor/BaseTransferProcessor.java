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

package com.webank.ai.fate.driver.federation.transfer.communication.processor;

import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.api.grpc.client.crud.ClusterMetaClient;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.fate.driver.federation.factory.TransferServiceFactory;
import com.webank.ai.fate.driver.federation.transfer.manager.TransferMetaHelper;
import com.webank.ai.fate.driver.federation.transfer.utils.TransferPojoUtils;
import com.webank.ai.fate.driver.federation.utils.FederationServerUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import javax.annotation.PostConstruct;

public abstract class BaseTransferProcessor implements Runnable {
    protected final Federation.TransferMeta transferMeta;
    @Autowired
    protected StorageMetaClient storageMetaClient;
    @Autowired
    protected ClusterMetaClient clusterMetaClient;
    @Autowired
    protected TransferMetaHelper transferMetaHelper;
    @Autowired
    protected TransferServiceFactory transferServiceFactory;
    @Autowired
    protected ThreadPoolTaskExecutor ioProducerPool;
    @Autowired
    protected ThreadPoolTaskExecutor ioConsumerPool;
    @Autowired
    protected ToStringUtils toStringUtils;
    @Autowired
    protected FederationServerUtils federationServerUtils;
    @Autowired
    protected TransferPojoUtils transferPojoUtils;
    @Autowired
    protected ErrorUtils errorUtils;

    protected String transferMetaId;

    private final Logger LOGGER = LogManager.getLogger(this);

    public BaseTransferProcessor(Federation.TransferMeta transferMeta) {
        this.transferMeta = transferMeta;
    }

    @PostConstruct
    public void init() {
        // todo: init with parameters
        try {
            LOGGER.info("[FEDERATION][BASE][PROCESSOR] init");
            storageMetaClient.init(federationServerUtils.getMetaServiceEndpoint());
            clusterMetaClient.init(federationServerUtils.getMetaServiceEndpoint());

            transferMetaId = transferPojoUtils.generateTransferId(transferMeta);
        } catch (Exception e) {
            LOGGER.error(errorUtils.getStackTrace(e));
        }
    }
}
