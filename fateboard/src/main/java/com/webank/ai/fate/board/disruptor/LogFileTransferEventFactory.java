package com.webank.ai.fate.board.disruptor;

import com.lmax.disruptor.EventFactory;

public class LogFileTransferEventFactory implements EventFactory<LogFileTransferEvent> {
    @Override
    public LogFileTransferEvent newInstance() {
        return new LogFileTransferEvent();
    }
}