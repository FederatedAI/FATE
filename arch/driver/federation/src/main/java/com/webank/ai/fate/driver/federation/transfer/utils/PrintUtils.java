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

package com.webank.ai.fate.driver.federation.transfer.utils;

import com.webank.ai.fate.driver.federation.transfer.manager.RecvBrokerManager;
import com.webank.ai.fate.driver.federation.transfer.model.TransferBroker;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;

@Component
@Scope("prototype")
public class PrintUtils {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RecvBrokerManager recvBrokerManager;

    public void printTransferBrokerTemporaryHolder(String... prefix) {
        LOGGER.info("--- start print ---");

        int count = 0;
        for (String s : prefix) {
            LOGGER.info("--- prefix {}: {} ---", ++count, s);
        }

        Set<Map.Entry<String, TransferBroker>> entries = recvBrokerManager.getEntrySet();
        for (Map.Entry<String, TransferBroker> entry : entries) {
            LOGGER.info("--- entry: {} : {}", entry.getKey(), entry.getValue());
        }

        LOGGER.info("total size: {}", entries.size());
        LOGGER.info("--- end print ---");
    }
}
