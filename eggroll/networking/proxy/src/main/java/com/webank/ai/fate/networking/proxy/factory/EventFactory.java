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

package com.webank.ai.fate.networking.proxy.factory;


import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.event.model.PipeHandleNotificationEvent;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.model.PipeHandlerInfo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class EventFactory {
    @Autowired
    private LocalBeanFactory localBeanFactory;

    public PipeHandleNotificationEvent createPipeHandleNotificationEvent(Object source,
                                                                         PipeHandleNotificationEvent.Type type,
                                                                         Proxy.Metadata metadata,
                                                                         Pipe pipe) {
        PipeHandlerInfo pipeHandlerInfo = (PipeHandlerInfo) localBeanFactory
                .getBean(PipeHandlerInfo.class, type, metadata, pipe);

        return (PipeHandleNotificationEvent) localBeanFactory
                .getBean(PipeHandleNotificationEvent.class, source, pipeHandlerInfo);
    }

    public PipeHandleNotificationEvent createPipeHandleNotificationEvent(Object source,
                                                                         PipeHandleNotificationEvent.Type type,
                                                                         Proxy.Packet packet,
                                                                         Pipe pipe) {
        PipeHandlerInfo pipeHandlerInfo = (PipeHandlerInfo) localBeanFactory
                .getBean(PipeHandlerInfo.class, type, packet, pipe);

        return (PipeHandleNotificationEvent) localBeanFactory
                .getBean(PipeHandleNotificationEvent.class, source, pipeHandlerInfo);
    }
}
