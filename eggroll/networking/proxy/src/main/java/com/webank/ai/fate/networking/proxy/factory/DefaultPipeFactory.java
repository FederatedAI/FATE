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
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.infra.impl.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.InputStream;
import java.io.OutputStream;

@Component("defaultPipeFactory")
public class DefaultPipeFactory implements PipeFactory {
    @Autowired
    private LocalBeanFactory localBeanFactory;

    public InputStreamOutputStreamNoStoragePipe createInputStreamOutputStreamNoStoragePipe(InputStream is,
                                                                                           OutputStream os,
                                                                                           Proxy.Metadata metadata) {
        return (InputStreamOutputStreamNoStoragePipe) localBeanFactory
                .getBean(InputStreamOutputStreamNoStoragePipe.class, is, os, metadata);
    }

    public InputStreamToPacketUnidirectionalPipe createInputStreamToPacketUnidirectionalPipe(InputStream is,
                                                                                             Proxy.Metadata metadata) {
        return (InputStreamToPacketUnidirectionalPipe) localBeanFactory
                .getBean(InputStreamToPacketUnidirectionalPipe.class, is, metadata);
    }

    public InputStreamToPacketUnidirectionalPipe createInputStreamToPacketUnidirectionalPipe(InputStream is,
                                                                                             Proxy.Metadata metadata,
                                                                                             int trunkSize) {
        return (InputStreamToPacketUnidirectionalPipe) localBeanFactory
                .getBean(InputStreamToPacketUnidirectionalPipe.class, is, metadata, trunkSize);
    }

    public PacketToOutputStreamUnidirectionalPipe createPacketToOutputStreamUnidirectionalPipe(OutputStream os) {
        return (PacketToOutputStreamUnidirectionalPipe) localBeanFactory
                .getBean(PacketToOutputStreamUnidirectionalPipe.class, os);
    }

    public PacketQueuePipe createPacketQueuePipe(Proxy.Metadata metadata) {
        return (PacketQueuePipe) localBeanFactory.getBean(PacketQueuePipe.class, metadata);
    }

    @Override
    public Pipe create() {
        return (PacketQueueSingleResultPipe) localBeanFactory.getBean(PacketQueueSingleResultPipe.class);
    }
}
