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

package com.webank.ai.fate.driver.federation.transfer.api.grpc.observer;

import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.eggroll.core.api.grpc.observer.CallerWithSameTypeDelayedResultResponseStreamObserver;
import com.webank.ai.eggroll.core.model.DelayedResult;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class UnaryCallServerRequestStreamObserver extends CallerWithSameTypeDelayedResultResponseStreamObserver<Proxy.Packet, Proxy.Packet> {
    public UnaryCallServerRequestStreamObserver(CountDownLatch finishLatch, DelayedResult<Proxy.Packet> delayedResult) {
        super(finishLatch, delayedResult);
    }
}
