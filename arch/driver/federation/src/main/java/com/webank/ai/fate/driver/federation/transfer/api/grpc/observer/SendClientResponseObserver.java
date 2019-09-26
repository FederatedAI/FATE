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

import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.api.grpc.observer.CallerWithSameTypeDelayedResultResponseStreamObserver;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;

@Component
@Scope("prototype")
public class SendClientResponseObserver extends CallerWithSameTypeDelayedResultResponseStreamObserver<Federation.TransferMeta, Federation.TransferMeta> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ToStringUtils toStringUtils;

    public SendClientResponseObserver(CountDownLatch finishLatch, DelayedResult<Federation.TransferMeta> delayedResult) {
        super(finishLatch, delayedResult);
    }

    @Override
    public void onNext(Federation.TransferMeta transferMeta) {
        LOGGER.info("[SEND][CLIENT][OBSERVER]Send client response received: {}", toStringUtils.toOneLineString(transferMeta));
        super.onNext(transferMeta);
    }

    @Override
    public void onError(Throwable throwable) {
        LOGGER.info("[SEND][CLIENT][OBSERVER] onError");
        super.onError(throwable);
    }

    @Override
    public void onCompleted() {
        LOGGER.info("[SEND][CLIENT][OBSERVER] onCompleted");
        super.onCompleted();
    }
}
