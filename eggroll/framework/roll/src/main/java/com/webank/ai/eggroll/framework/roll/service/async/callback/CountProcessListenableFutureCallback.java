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

package com.webank.ai.eggroll.framework.roll.service.async.callback;

import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;

@Component
@Scope("prototype")
public class CountProcessListenableFutureCallback implements ListenableFutureCallback<Kv.Count> {
    private static final Logger LOGGER = LogManager.getLogger();
    private final AtomicLong countResult;
    private final List<Throwable> errorContainer;
    private final CountDownLatch finishLatch;
    private final String ip;
    private final int port;
    @Autowired
    private ErrorUtils errorUtils;

    public CountProcessListenableFutureCallback(AtomicLong countResult,
                                                List<Throwable> errorContainer,
                                                CountDownLatch finishLatch,
                                                String ip,
                                                int port) {
        this.countResult = countResult;
        this.errorContainer = errorContainer;
        this.finishLatch = finishLatch;
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void onFailure(Throwable ex) {
        LOGGER.warn("[ROLL][PROCESS][Count][Callback] {}:{}: error: {}", ip, port, errorUtils.getStackTrace(ex));
        errorContainer.add(ex);
        finishLatch.countDown();
    }

    @Override
    public void onSuccess(Kv.Count result) {
        if (result != null) {
            long resultValue = result.getValue();
            LOGGER.warn("[ROLL][PROCESS][Count][Callback] result value: {}, address : {}:{}", resultValue, ip, port);
            countResult.addAndGet(result.getValue());
            finishLatch.countDown();
        } else {
            LOGGER.warn("[ROLL][PROCESS][Count][Callback] result is null. address: {}:{}", ip, port);
        }
    }
}
