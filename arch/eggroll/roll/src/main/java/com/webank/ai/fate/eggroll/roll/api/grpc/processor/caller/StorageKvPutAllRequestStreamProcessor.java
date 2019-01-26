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
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Node;
import com.webank.ai.fate.eggroll.roll.service.model.OperandBroker;
import io.grpc.stub.StreamObserver;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
// todo: see if this can be merged with PushStreamProcessor
public class StorageKvPutAllRequestStreamProcessor extends BaseStreamProcessor<Kv.Operand> {
    private OperandBroker broker;
    private Node node;

    public StorageKvPutAllRequestStreamProcessor(StreamObserver<Kv.Operand> streamObserver, OperandBroker broker, Node node) {
        super(streamObserver);
        this.broker = broker;
        this.node = node;
    }

    @Override
    public void process() {
        List<Kv.Operand> operands = Lists.newLinkedList();
        broker.drainTo(operands);

        for (Kv.Operand operand : operands) {
            streamObserver.onNext(operand);
        }
    }
}
