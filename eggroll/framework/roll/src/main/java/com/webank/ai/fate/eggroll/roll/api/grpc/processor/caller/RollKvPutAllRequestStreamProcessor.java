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

package com.webank.ai.fate.eggroll.roll.api.grpc.processor.caller;

import com.google.common.collect.Lists;
import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.core.api.grpc.client.crud.BaseStreamProcessor;
import com.webank.ai.fate.eggroll.roll.service.model.OperandBroker;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
public class RollKvPutAllRequestStreamProcessor
        extends BaseStreamProcessor<Kv.Operand> {
    private OperandBroker operandBroker;
    private List<Kv.Operand> operandsToSend;
    private int entryCount = 0;
    private static final Logger LOGGER = LogManager.getLogger();

    public RollKvPutAllRequestStreamProcessor(StreamObserver<Kv.Operand> streamObserver, OperandBroker operandBroker) {
        super(streamObserver);
        this.operandBroker = operandBroker;
        this.operandsToSend = Lists.newLinkedList();
    }

    @Override
    public void process() {
        super.process();
        operandsToSend = Lists.newLinkedList();
        operandBroker.drainTo(operandsToSend, 500_000);
        for (Kv.Operand operand : operandsToSend) {
            streamObserver.onNext(operand);
            ++entryCount;
        }
    }

    @Override
    public void complete() {
        // operandBroker.setFinished();
        // LOGGER.info("[ROLL][PUTALL][COMPLETE] trying to complete putAll stream processor. remaining: {}, total entryCount: {}", operandBroker.getQueueSize(), entryCount);

/*        while (!operandBroker.isClosable()) {
            process();
        }*/

        LOGGER.info("[ROLL][PUTALL][COMPLETE] actual completes putAll stream processor for this batch. remaining: {}, total entryCount: {}", operandBroker.getQueueSize(), entryCount);

        super.complete();
    }
}
