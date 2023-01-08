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
package com.osx.core.queue;

import com.osx.core.constant.TransferStatus;


public class TranferQueueInfo {
    String transferId;
    TransferStatus transferStatus;
    long createTimestamp;
    long lastStatusChangeTimestamp;
    long lastWriteTimestamp;
    long lastReadTimestamp;
    long logicOffset;

    public long getLogicOffset() {
        return logicOffset;
    }

    public void setLogicOffset(long logicOffset) {
        this.logicOffset = logicOffset;
    }

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
