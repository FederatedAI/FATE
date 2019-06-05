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

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class PutAllProcessorListenableFutureCallback implements ListenableFutureCallback<BasicMeta.ReturnStatus> {
    protected final List<BasicMeta.ReturnStatus> results;
    protected final List<Throwable> errorContainer;
    protected final CountDownLatch finishLatch;
    protected final String ip;
    protected final int port;

    @Autowired
    protected ErrorUtils errorUtils;

    private final StoreInfo storeInfoWithFragment;
    private Set<Integer> finishedFragments;

    private static final Logger LOGGER = LogManager.getLogger();

    public PutAllProcessorListenableFutureCallback(List<BasicMeta.ReturnStatus> results,
                                                   List<Throwable> errorContainer,
                                                   CountDownLatch finishLatch,
                                                   String ip,
                                                   int port,
                                                   StoreInfo storeInfoWithFragment,
                                                   Set<Integer> finishedFragments) {

        this.results = results;
        this.errorContainer = errorContainer;
        this.finishLatch = finishLatch;
        this.ip = ip;
        this.port = port;
        this.storeInfoWithFragment = storeInfoWithFragment;
        this.finishedFragments = finishedFragments;
    }

    @Override
    public void onFailure(Throwable throwable) {
        LOGGER.error("[ROLL][KV][PUTALL][ONFAILURE] storeInfo: {}, latch count: {}, {}:{}: error: {}",
                storeInfoWithFragment, finishLatch.getCount(), ip, port, errorUtils.getStackTrace(throwable));

        errorContainer.add(throwable);
        finishedFragments.add(storeInfoWithFragment.getFragment());
        finishLatch.countDown();
    }

    @Override
    public void onSuccess(BasicMeta.ReturnStatus result) {
        LOGGER.info("[ROLL][KV][PUTALL][ONSUCCESS] put all success. storeInfo: {}, latch count: {}, address: {}:{}",
                storeInfoWithFragment, finishLatch.getCount(), ip, port);
        results.add(result);

        finishedFragments.add(storeInfoWithFragment.getFragment());
        finishLatch.countDown();
    }
}
