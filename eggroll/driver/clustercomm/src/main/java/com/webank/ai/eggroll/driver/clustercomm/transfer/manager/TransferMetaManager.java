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

package com.webank.ai.eggroll.driver.clustercomm.transfer.manager;

import com.google.common.collect.Maps;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.driver.clustercomm.transfer.event.TransferJobEvent;
import com.webank.ai.eggroll.driver.clustercomm.transfer.utils.TransferPojoUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
class TransferMetaManager {
    private static final Logger LOGGER = LogManager.getLogger();
    private final Map<String, ClusterComm.TransferMeta> transferMetaHolder;
    @Autowired
    private ApplicationEventPublisher applicationEventPublisher;
    @Autowired
    private TransferPojoUtils transferPojoUtils;
    @Autowired
    private ToStringUtils toStringUtils;

    public TransferMetaManager() {
        this.transferMetaHolder = Maps.newConcurrentMap();
    }

    synchronized void create(ClusterComm.TransferMeta transferMeta) {
        String key = transferPojoUtils.generateTransferId(transferMeta);

        ClusterComm.TransferMeta existing = get(transferMeta);
        if (existing == null) {
            transferMetaHolder.put(key, transferMeta);
        }
        applicationEventPublisher.publishEvent(new TransferJobEvent(this, transferMeta));
    }

    public ClusterComm.TransferMeta get(ClusterComm.TransferMeta transferMeta) {
        String key = transferPojoUtils.generateTransferId(transferMeta);

        // LOGGER.info("[TransferMeta] getting transferMeta for key: {}", key);

        return this.get(key);
    }

    public ClusterComm.TransferMeta get(String key) {
        return transferMetaHolder.get(key);
    }

    boolean update(ClusterComm.TransferMeta transferMeta) {
        boolean result = false;
        String key = transferPojoUtils.generateTransferId(transferMeta);

        if (transferMetaHolder.containsKey(key)) {
            transferMetaHolder.put(key, transferMeta);
            result = true;
        }

        return result;
    }
}
