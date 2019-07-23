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