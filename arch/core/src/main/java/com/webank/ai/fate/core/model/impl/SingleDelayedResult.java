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

package com.webank.ai.fate.core.model.impl;

import com.webank.ai.fate.core.model.DelayedResult;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Service;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

@Service(value = "resultCallback")
@Scope("prototype")
public class SingleDelayedResult<T> implements DelayedResult<T> {
    private final CountDownLatch finishLatch;
    private T result;
    private volatile Throwable throwable;
    private volatile boolean hasResultBeenSet;

    public SingleDelayedResult() {
        this.finishLatch = new CountDownLatch(1);
        this.hasResultBeenSet = false;
        this.throwable = null;
    }

    @Override
    public T getResultNow() {
        if (!hasResultBeenSet) {
            throw new IllegalStateException("no result has been set yet");
        }
        return result;
    }

    @Override
    public T getResult(long timeout, TimeUnit timeUnit) throws InterruptedException {
        boolean waitResult = finishLatch.await(timeout, timeUnit);

        return result;
    }

    @Override
    public void setResult(T t) {
        result = t;
        hasResultBeenSet = true;
        finishLatch.countDown();
    }

    @Override
    public boolean hasResult() {
        return hasResultBeenSet;
    }

    @Override
    public boolean hasError() {
        return this.throwable != null;
    }

    @Override
    public Throwable getError() {
        return throwable;
    }

    @Override
    public void setError(Throwable throwable) {
        this.throwable = throwable;
        finishLatch.countDown();
    }

    @Override
    public final CountDownLatch getLatch() {
        return finishLatch;
    }

}
