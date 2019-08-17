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

import com.webank.ai.fate.api.networking.proxy.Proxy;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class PacketQueueSingleResultPipe extends PacketQueuePipe {
    private static final Logger LOGGER = LogManager.getLogger(PacketQueueSingleResultPipe.class);
    private Proxy.Metadata result;

    public PacketQueueSingleResultPipe() {
        this(null);
    }

    public PacketQueueSingleResultPipe(Proxy.Metadata metadata) {
        super(metadata);
        // this.queue = queueFactory.createConcurrentLinkedQueue();
    }

    public Proxy.Metadata getResult() {
        return result;
    }

    public void setResult(Proxy.Metadata metadata) {
        if (hasResult()) {
            throw new IllegalStateException("result has been set");
        }
        this.result = metadata;
    }

    public boolean hasResult() {
        return result != null;
    }
}
