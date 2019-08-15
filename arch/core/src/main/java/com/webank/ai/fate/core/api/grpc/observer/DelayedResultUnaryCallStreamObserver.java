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

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.core.model.DelayedResult;
import com.webank.ai.fate.core.utils.ErrorUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class DelayedResultUnaryCallStreamObserver
        extends BaseCallerResponseStreamObserver<BasicMeta.CallRequest, BasicMeta.CallResponse> {
    private static final Logger LOGGER = LogManager.getLogger(DelayedResultUnaryCallStreamObserver.class);
    @Autowired
    private ErrorUtils errorUtils;
    private DelayedResult<BasicMeta.CallResponse> delayedResult;

    public DelayedResultUnaryCallStreamObserver(DelayedResult<BasicMeta.CallResponse> delayedResult) {
        super(delayedResult.getLatch());
        this.delayedResult = delayedResult;
    }

    @Override
    public void onNext(BasicMeta.CallResponse e) {
        this.delayedResult.setResult(e);
    }

    @Override
    public void onError(Throwable throwable) {
        this.delayedResult.setError(throwable);
        super.onError(throwable);
    }

    @Override
    public void onCompleted() {
        super.onCompleted();
    }
}
