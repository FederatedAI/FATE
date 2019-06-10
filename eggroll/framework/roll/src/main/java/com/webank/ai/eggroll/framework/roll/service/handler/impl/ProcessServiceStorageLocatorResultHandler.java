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

package com.webank.ai.eggroll.framework.roll.service.handler.impl;

import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.roll.service.handler.ProcessServiceResultHandler;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
public class ProcessServiceStorageLocatorResultHandler implements ProcessServiceResultHandler<StorageBasic.StorageLocator, StorageBasic.StorageLocator> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ToStringUtils toStringUtils;

    @Override
    public void handle(StreamObserver<StorageBasic.StorageLocator> requestObserver, List<StorageBasic.StorageLocator> unprocessedResults) {
        StorageBasic.StorageLocator storageLocator = unprocessedResults.get(0);
        LOGGER.info("[HANDLE][RESULT] {}, unprocessedResults.size: {}", toStringUtils.toOneLineString(storageLocator), unprocessedResults.size());

        if (storageLocator == null) {
            for (StorageBasic.StorageLocator cur : unprocessedResults) {
                LOGGER.warn("[HANDLE][RESULT] cur: {}", toStringUtils.toOneLineString(cur));
            }
        }
        requestObserver.onNext(storageLocator);
    }
}
