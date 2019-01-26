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

package com.webank.ai.fate.networking.proxy.event.listener;

import com.webank.ai.fate.networking.proxy.event.model.PipeHandleNotificationEvent;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.infra.impl.PacketQueuePipe;
import com.webank.ai.fate.networking.proxy.model.PipeHandlerInfo;
import com.webank.ai.fate.networking.proxy.service.CascadedCaller;
import com.webank.ai.fate.networking.proxy.util.ToStringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationListener;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class PipeHandleNotificationEventListener implements ApplicationListener<PipeHandleNotificationEvent> {
    private static final Logger LOGGER = LogManager.getLogger(PipeHandleNotificationEventListener.class);
    @Autowired
    private ApplicationContext applicationContext;
    @Autowired
    private ToStringUtils toStringUtils;

    @Override
    public void onApplicationEvent(PipeHandleNotificationEvent pipeHandleNotificationEvent) {
        // LOGGER.warn("event listened: {}", pipeHandleNotificationEvent.getPipeHandlerInfo());
        LOGGER.info("event metadata: {}", toStringUtils.toOneLineString(pipeHandleNotificationEvent.getPipeHandlerInfo().getMetadata()));

        PipeHandlerInfo pipeHandlerInfo = pipeHandleNotificationEvent.getPipeHandlerInfo();
        Pipe pipe = pipeHandlerInfo.getPipe();

        if (pipe instanceof PacketQueuePipe) {
            CascadedCaller cascadedCaller = applicationContext.getBean(CascadedCaller.class, pipeHandlerInfo);
            cascadedCaller.run();
        }
    }
}
