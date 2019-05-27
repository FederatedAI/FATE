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

package com.webank.ai.fate.core.api.grpc.observer;

import com.webank.ai.fate.core.model.DelayedResult;

import java.util.concurrent.CountDownLatch;

/**
 * @param <R> calleR parameter type
 * @param <E> calleE parameter type
 * @param <D> DelayedResult parameter type
 */
public abstract class BaseCallerWithDelayedResultResponseStreamObserver<R, E, D> extends BaseCallerResponseStreamObserver<R, E> {

    // need to be set in derived class
    protected DelayedResult<D> delayedResult;

    public BaseCallerWithDelayedResultResponseStreamObserver(CountDownLatch finishLatch, DelayedResult<D> delayedResult) {
        super(finishLatch);
        this.delayedResult = delayedResult;
    }

    @Override
    public void onError(Throwable throwable) {
        delayedResult.setError(throwable);
        super.onError(throwable);
    }
}
