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

package com.webank.ai.eggroll.core.api.grpc.observer;

import io.grpc.stub.ClientCallStreamObserver;
import io.grpc.stub.ClientResponseObserver;

import java.util.concurrent.CountDownLatch;

public abstract class BaseFlowControlCallerResponseStreamObserver<R, E>
        extends BaseCallerResponseStreamObserver<R, E>
        implements ClientResponseObserver<R, E> {
    protected ClientCallStreamObserver<R> callerStreamObserver;

    public BaseFlowControlCallerResponseStreamObserver(CountDownLatch finishLatch) {
        super(finishLatch);
    }

    @Override
    public void beforeStart(ClientCallStreamObserver<R> callerStreamObserver) {
        this.callerStreamObserver = callerStreamObserver;
        callerStreamObserver.disableAutoInboundFlowControl();

        callerStreamObserver.setOnReadyHandler(() -> {
            while (callerStreamObserver.isReady()) {

            }
        });
    }
}
