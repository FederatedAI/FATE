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

package com.webank.ai.eggroll.framework.roll.api.grpc.processor.caller;

import com.google.common.collect.Lists;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.api.grpc.client.crud.BaseStreamProcessor;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
// todo: see if this can be merged with PushStreamProcessor
public class StorageKvPutAllRequestStreamProcessor extends BaseStreamProcessor<Kv.Operand> {
    private OperandBroker operandBroker;
    private Node node;
    private int entryCount = 0;
    private static final Logger LOGGER = LogManager.getLogger();

    public StorageKvPutAllRequestStreamProcessor(StreamObserver<Kv.Operand> streamObserver, OperandBroker operandBroker, Node node) {
        super(streamObserver);
        this.operandBroker = operandBroker;
        this.node = node;
    }

    @Override
    public synchronized void process() {
        super.process();
        List<Kv.Operand> operands = Lists.newLinkedList();
        operandBroker.drainTo(operands, 200_000);

        for (Kv.Operand operand : operands) {
            streamObserver.onNext(operand);
            ++entryCount;
        }
    }

    @Override
    public void complete() {
        // LOGGER.info("[PUTALL][SUBTASK] trying to complete putAll sub task. remaining: {}, entryCount: {}", operandBroker.getQueueSize(), entryCount);
/*        while (!broker.isClosable()) {
            process();
        }*/
        LOGGER.info("[PUTALL][SUBTASK] actual completes putAll sub task. remaining: {}, entryCount: {}", operandBroker.getQueueSize(), entryCount);
        super.complete();
    }

}
