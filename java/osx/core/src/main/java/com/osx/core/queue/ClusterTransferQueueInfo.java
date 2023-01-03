package com.osx.core.queue;


import com.osx.core.utils.JsonUtil;

public class ClusterTransferQueueInfo {

    String tranferId;
    String ip;
    int port;
    long createTimestamp;
    public ClusterTransferQueueInfo() {

    }

    public String getTranferId() {
        return tranferId;
    }

    public void setTranferId(String tranferId) {
        this.tranferId = tranferId;
    }

    public String getIp() {
        return ip;
    }

    public void setIp(String ip) {
        this.ip = ip;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public long getCreateTimestamp() {
        return createTimestamp;
    }

    public void setCreateTimestamp(long createTimestamp) {
        this.createTimestamp = createTimestamp;
    }

    public String toString() {
        return JsonUtil.object2Json(this);
    }

}