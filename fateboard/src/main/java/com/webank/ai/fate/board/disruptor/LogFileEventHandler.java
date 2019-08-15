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

import com.lmax.disruptor.EventHandler;
import com.webank.ai.fate.board.ssh.SftpUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.List;

@Component
public class LogFileEventHandler implements EventHandler<LogFileTransferEvent> {

    Logger logger = LoggerFactory.getLogger(LogFileEventHandler.class);

    @Override
    public void onEvent(LogFileTransferEvent event, long sequence, boolean endOfBatch) {

        logger.info("receive transfer event {}", event);
        File file = new File(event.desFilePath);
        if (file.exists()) {
            return;
        }

        List<String> files = SftpUtils.batchDownLoadFile(event.sshInfo, event.sourceFilePath,
                event.desFilePath, null, ".log", false);

    }
}