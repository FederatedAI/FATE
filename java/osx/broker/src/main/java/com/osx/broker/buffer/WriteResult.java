package com.osx.broker.buffer;


import com.osx.core.utils.JsonUtil;

public class WriteResult {
    WriteStatus status;
    int dataSize;
    int writeIndex;

    public WriteResult(WriteStatus status, int dataSize, int writeIndex) {
        this.status = status;
        this.dataSize = dataSize;
        this.writeIndex = writeIndex;
    }

    public WriteStatus getStatus() {
        return status;
    }

    public void setStatus(WriteStatus status) {
        this.status = status;
    }

    public int getDataSize() {
        return dataSize;
    }

    public void setDataSize(int dataSize) {
        this.dataSize = dataSize;
    }

    public int getWriteIndex() {
        return writeIndex;
    }

    public void setWriteIndex(int writeIndex) {
        this.writeIndex = writeIndex;
    }

    public String toString() {
        return JsonUtil.object2Json(this);
    }
}