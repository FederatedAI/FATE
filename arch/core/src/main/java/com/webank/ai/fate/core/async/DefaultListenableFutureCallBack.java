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

package com.webank.ai.fate.core.async;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.List;
import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class DefaultListenableFutureCallBack<T> implements ListenableFutureCallback<T> {
    private static final Logger LOGGER = LogManager.getLogger();
    private final List<Throwable> throwables;
    private final CountDownLatch finishLatch;

    public DefaultListenableFutureCallBack(List<Throwable> throwables, CountDownLatch finishLatch) {
        this.throwables = throwables;
        this.finishLatch = finishLatch;
    }

    @Override
    public void onFailure(Throwable ex) {
        this.throwables.add(ex);
        finishLatch.countDown();
    }

    @Override
    public void onSuccess(T result) {
        finishLatch.countDown();
    }
}
