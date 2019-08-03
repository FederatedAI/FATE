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
package com.webank.ai.fate.board.disruptor;

import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.util.DaemonThreadFactory;
import com.webank.ai.fate.board.pojo.SshInfo;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class LogFileTransferEventProducer implements InitializingBean {
    @Autowired
    LogFileEventHandler logFileEventHandle;
    private RingBuffer<LogFileTransferEvent> ringBuffer;

    public void onData(SshInfo sshInfo,
                       String sourceFilePath,
                       String desFilePath) {
        long sequence = ringBuffer.next();  // Grab the next sequence
        try {
            LogFileTransferEvent event = ringBuffer.get(sequence); // Get the entry in the Disruptor
            event.setDesFilePath(desFilePath);
            event.setSourceFilePath(sourceFilePath);
            event.setSshInfo(sshInfo);

        } finally {
            ringBuffer.publish(sequence);
        }
    }

    @Override
    public void afterPropertiesSet() throws Exception {

        LogFileTransferEventFactory factory = new LogFileTransferEventFactory();
        int bufferSize = 1024;
        Disruptor<LogFileTransferEvent> disruptor = new Disruptor<>(factory, bufferSize, DaemonThreadFactory.INSTANCE);
        disruptor.handleEventsWith(logFileEventHandle);
        disruptor.start();
        this.ringBuffer = disruptor.getRingBuffer();
    }
}