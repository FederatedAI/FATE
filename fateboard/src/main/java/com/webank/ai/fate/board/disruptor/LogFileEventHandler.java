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