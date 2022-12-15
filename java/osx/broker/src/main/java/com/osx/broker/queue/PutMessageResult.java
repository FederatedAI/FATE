
package com.osx.broker.queue;

import com.osx.broker.message.AppendMessageResult;

public class PutMessageResult {
    private PutMessageStatus putMessageStatus;
    private AppendMessageResult appendMessageResult;

    public long getMsgLogicOffset() {
        return msgLogicOffset;
    }

    public void setMsgLogicOffset(long msgLogicOffset) {
        this.msgLogicOffset = msgLogicOffset;
    }

    private long  msgLogicOffset;


    public PutMessageResult(PutMessageStatus putMessageStatus, AppendMessageResult appendMessageResult) {
        this.putMessageStatus = putMessageStatus;
        this.appendMessageResult = appendMessageResult;
    }

    public boolean isOk() {
        return this.appendMessageResult != null && this.appendMessageResult.isOk();
    }

    public AppendMessageResult getAppendMessageResult() {
        return appendMessageResult;
    }

    public void setAppendMessageResult(AppendMessageResult appendMessageResult) {
        this.appendMessageResult = appendMessageResult;
    }

    public PutMessageStatus getPutMessageStatus() {
        return putMessageStatus;
    }

    public void setPutMessageStatus(PutMessageStatus putMessageStatus) {
        this.putMessageStatus = putMessageStatus;
    }

    @Override
    public String toString() {
        return "PutMessageResult [putMessageStatus=" + putMessageStatus + ", appendMessageResult="
            + appendMessageResult + "]";
    }

}
