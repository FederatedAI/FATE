
package com.osx.broker.message;

/**
 * When write a message to the commit log, returns results
 */
public class AppendMessageResult {
    // Return code
    private AppendMessageStatus status;
    // Where to start writing
    private long wroteOffset;
    // Write Bytes
    private int wroteBytes;
    // Message ID
    private String msgId;
    // Message storage timestamp
    private long storeTimestamp;
    // Consume queue's offset(step by one)
    private long logicsOffset;
    private long pagecacheRT = 0;

    private int msgNum = 1;

    public AppendMessageResult(AppendMessageStatus status) {
        this(status, 0, 0, "", 0, 0, 0);
    }

    public AppendMessageResult(AppendMessageStatus status, long wroteOffset, int wroteBytes, String msgId,
        long storeTimestamp, long logicsOffset, long pagecacheRT) {
        this.status = status;
        this.wroteOffset = wroteOffset;
        this.wroteBytes = wroteBytes;
        this.msgId = msgId;
        this.storeTimestamp = storeTimestamp;
        this.logicsOffset = logicsOffset;
        this.pagecacheRT = pagecacheRT;
    }

    public long getPagecacheRT() {
        return pagecacheRT;
    }

    public void setPagecacheRT(final long pagecacheRT) {
        this.pagecacheRT = pagecacheRT;
    }

    public boolean isOk() {
        return this.status == AppendMessageStatus.PUT_OK;
    }

    public AppendMessageStatus getStatus() {
        return status;
    }

    public void setStatus(AppendMessageStatus status) {
        this.status = status;
    }

    public long getWroteOffset() {
        return wroteOffset;
    }

    public void setWroteOffset(long wroteOffset) {
        this.wroteOffset = wroteOffset;
    }

    public int getWroteBytes() {
        return wroteBytes;
    }

    public void setWroteBytes(int wroteBytes) {
        this.wroteBytes = wroteBytes;
    }

    public String getMsgId() {
        return msgId;
    }

    public void setMsgId(String msgId) {
        this.msgId = msgId;
    }

    public long getStoreTimestamp() {
        return storeTimestamp;
    }

    public void setStoreTimestamp(long storeTimestamp) {
        this.storeTimestamp = storeTimestamp;
    }

    public long getLogicsOffset() {
        return logicsOffset;
    }

    public void setLogicsOffset(long logicsOffset) {
        this.logicsOffset = logicsOffset;
    }

    public int getMsgNum() {
        return msgNum;
    }

    public void setMsgNum(int msgNum) {
        this.msgNum = msgNum;
    }

    @Override
    public String toString() {
        return "AppendMessageResult{" +
            "status=" + status +
            ", wroteOffset=" + wroteOffset +
            ", wroteBytes=" + wroteBytes +
            ", msgId='" + msgId + '\'' +
            ", storeTimestamp=" + storeTimestamp +
            ", logicsOffset=" + logicsOffset +
            ", pagecacheRT=" + pagecacheRT +
            ", msgNum=" + msgNum +
            '}';
    }
}
