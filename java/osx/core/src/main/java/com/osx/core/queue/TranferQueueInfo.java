package com.osx.core.queue;

import com.osx.core.constant.TransferStatus;


public class TranferQueueInfo {
    String transferId;
    TransferStatus transferStatus;
    long createTimestamp;
    long lastStatusChangeTimestamp;
    long lastWriteTimestamp;
    long lastReadTimestamp;

    public long getLogicOffset() {
        return logicOffset;
    }

    public void setLogicOffset(long logicOffset) {
        this.logicOffset = logicOffset;
    }

    long logicOffset;

    public String getTransferId() {
        return transferId;
    }

    public void setTransferId(String transferId) {
        this.transferId = transferId;
    }

    public TransferStatus getTransferStatus() {
        return transferStatus;
    }

    public void setTransferStatus(TransferStatus transferStatus) {
        this.transferStatus = transferStatus;
    }

    public long getCreateTimestamp() {
        return createTimestamp;
    }

    public void setCreateTimestamp(long createTimestamp) {
        this.createTimestamp = createTimestamp;
    }

    public long getLastStatusChangeTimestamp() {
        return lastStatusChangeTimestamp;
    }

    public void setLastStatusChangeTimestamp(long lastStatusChangeTimestamp) {
        this.lastStatusChangeTimestamp = lastStatusChangeTimestamp;
    }

    public long getLastWriteTimestamp() {
        return lastWriteTimestamp;
    }

    public void setLastWriteTimestamp(long lastWriteTimestamp) {
        this.lastWriteTimestamp = lastWriteTimestamp;
    }

    public long getLastReadTimestamp() {
        return lastReadTimestamp;
    }

    public void setLastReadTimestamp(long lastReadTimestamp) {
        this.lastReadTimestamp = lastReadTimestamp;
    }
}
