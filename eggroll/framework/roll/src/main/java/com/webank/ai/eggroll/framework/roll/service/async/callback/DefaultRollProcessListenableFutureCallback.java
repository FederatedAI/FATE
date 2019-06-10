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

import com.webank.ai.eggroll.core.utils.ErrorUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.List;
import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class DefaultRollProcessListenableFutureCallback<T> implements ListenableFutureCallback<T> {
    private static final Logger LOGGER = LogManager.getLogger();
    protected final List<T> results;
    protected final List<Throwable> errorContainer;
    protected final CountDownLatch finishLatch;
    protected final String ip;
    protected final int port;

    @Autowired
    protected ErrorUtils errorUtils;

    public DefaultRollProcessListenableFutureCallback(List<T> results,
                                                      List<Throwable> errorContainer,
                                                      CountDownLatch finishLatch,
                                                      String ip,
                                                      int port) {
        this.results = results;
        this.errorContainer = errorContainer;
        this.finishLatch = finishLatch;
        this.ip = ip;
        this.port = port;
    }

    @Override
    public void onFailure(Throwable ex) {
        LOGGER.error("[CALLBACK][ONFAILURE] {}:{}: error: {}", ip, port, errorUtils.getStackTrace(ex));
        errorContainer.add(ex);
        finishLatch.countDown();
    }

    @Override
    public void onSuccess(T result) {
        if (result == null) {
            LOGGER.error("[CALLBACK][ONSUCCESS] but null result. address: {}:{}", ip, port);
        }
        results.add(result);
        finishLatch.countDown();
    }
}
