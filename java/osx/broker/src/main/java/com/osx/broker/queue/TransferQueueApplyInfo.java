package com.osx.broker.queue;


import com.osx.core.utils.JsonUtil;

public class TransferQueueApplyInfo {

    String transferId;
    String instanceId;
    String ip;
    String sessionId;
    long applyTimestamp;

    public String getTransferId() {
        return transferId;
    }

    public void setTransferId(String transferId) {
        this.transferId = transferId;
    }

    public String getInstanceId() {
        return instanceId;
    }

    public void setInstanceId(String instanceId) {
        this.instanceId = instanceId;
    }

    public String getIp() {
        return ip;
    }

    public void setIp(String ip) {
        this.ip = ip;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public long getApplyTimestamp() {
        return applyTimestamp;
    }

    public void setApplyTimestamp(long applyTimestamp) {
        this.applyTimestamp = applyTimestamp;
    }

    public String toString() {
        return JsonUtil.object2Json(this);
    }

}