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

package com.webank.ai.eggroll.framework.roll.api.grpc.observer.processor.egg;

import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.api.grpc.observer.BaseCallerResponseStreamObserver;
import com.webank.ai.eggroll.framework.roll.service.model.OperandBroker;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class EggProcessorReduceResponseStreamObserver extends BaseCallerResponseStreamObserver<Processor.UnaryProcess, Kv.Operand> {
    private OperandBroker operandBroker;

    public EggProcessorReduceResponseStreamObserver(CountDownLatch finishLatch, OperandBroker operandBroker) {
        super(finishLatch);
        this.operandBroker = operandBroker;
    }

    @Override
    public void onNext(Kv.Operand operand) {
        operandBroker.put(operand);
    }

    @Override
    public void onError(Throwable throwable) {
        operandBroker.setFinished();
        super.onError(throwable);
    }

    @Override
    public void onCompleted() {
        operandBroker.setFinished();
        super.onCompleted();
    }
}
