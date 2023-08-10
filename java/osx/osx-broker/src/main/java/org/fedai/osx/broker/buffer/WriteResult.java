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
package org.fedai.osx.broker.buffer;


import org.fedai.osx.core.utils.JsonUtil;

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