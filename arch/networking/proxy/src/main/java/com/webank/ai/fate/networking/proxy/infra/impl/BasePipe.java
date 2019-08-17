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

package com.webank.ai.fate.networking.proxy.infra.impl;

import com.webank.ai.fate.networking.proxy.infra.Pipe;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public abstract class BasePipe implements Pipe {
    private final CountDownLatch closeLatch;
    private volatile boolean closed = false;
    private volatile boolean drained = false;
    private volatile Throwable throwable = null;

    public BasePipe() {
        this.closeLatch = new CountDownLatch(1);
    }

    @Override
    public Object read() {
        throw new UnsupportedOperationException("Operation not implemented");
    }

    @Override
    public void write(Object o) {
        throw new UnsupportedOperationException("Operation not implemented");
    }

    @Override
    public void onError(Throwable t) {
        setDrained();
        close();
        this.throwable = t;
    }

    @Override
    public void onComplete() {
        // setDrained();
        close();
    }

    @Override
    public synchronized void close() {
        setDrained();
        closeLatch.countDown();
        this.closed = true;
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void setDrained() {
        this.drained = true;
    }

    @Override
    public boolean isDrained() {
        return this.drained;
    }

    @Override
    public void awaitClosed() throws InterruptedException {
        closeLatch.await();
    }

    @Override
    public void awaitClosed(long timeout, TimeUnit unit) throws InterruptedException {
        closeLatch.await(timeout, unit);
    }

    @Override
    public boolean hasError() {
        return throwable != null;
    }

    @Override
    public Throwable getError() {
        return throwable;
    }

    public void checkNotClosed() {
        if (isClosed()) {
            throw new IllegalStateException("pipe closed");
        }
    }
}
