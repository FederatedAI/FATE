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

import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.roll.factory.RollModelFactory;
import com.webank.ai.eggroll.framework.roll.service.handler.ProcessServiceResultHandler;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBrokerUnSortedHub;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
public class ReduceProcessServiceResultHandler implements ProcessServiceResultHandler<OperandBroker, Kv.Operand> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private RollModelFactory rollModelFactory;
    @Autowired
    private ThreadPoolTaskExecutor asyncThreadPool;
    @Autowired
    private ToStringUtils toStringUtils;

    @Override
    public void handle(StreamObserver<Kv.Operand> requestObserver, List<OperandBroker> unprocessedResults) {
        OperandBrokerUnSortedHub operandBrokerUnsortedHub = rollModelFactory.createOperandUnsortedBrokerHub();
        LOGGER.info("[REDUCE][HANDLER] unprocessedResult.size: {}", unprocessedResults.size());
        for (OperandBroker operandBroker : unprocessedResults) {
            // LOGGER.info("operandBroker: {}", operandBroker);
            operandBrokerUnsortedHub.addBranchBroker(operandBroker);
        }

        asyncThreadPool.submit(operandBrokerUnsortedHub);

        OperandBroker mergedOperandBroker = operandBrokerUnsortedHub.getMergedBroker();
        Kv.Operand result = null;

        int resultCount = 0;
        int resultNotNullCount = 0;
        // todo: to eden: please add result process logic here
        while (!mergedOperandBroker.isClosable()) {
            result = mergedOperandBroker.get();
            ++resultCount;
            if (result != null) {
                // LOGGER.info("[REDUCE][RESULT]: {}", toStringUtils.toOneLineString(result));
                requestObserver.onNext(result);
                ++resultNotNullCount;
            }
        }

        LOGGER.info("[REDUCE][COMPLETE] done handling reduce results. result count: {}, result not null count: {}", resultCount, resultNotNullCount);
    }
}
