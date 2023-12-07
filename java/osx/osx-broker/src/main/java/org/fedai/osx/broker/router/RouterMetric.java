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
package org.fedai.osx.broker.router;

import org.fedai.osx.core.utils.JsonUtil;

import java.util.concurrent.atomic.AtomicLong;

public class RouterMetric {

    long lastCheckTimestamp;
    AtomicLong sourceReceiveBytesCount = new AtomicLong(0);
    AtomicLong sourceSendBytesCount = new AtomicLong(0);
    AtomicLong sinkReceiveBytesCount = new AtomicLong(0);
    AtomicLong sinkSendBytesCount = new AtomicLong(0);
    long lastUpstreamBytesCount;
    long lastDownstreamBytesCount;

    public static void main(String[] args) {
        RouterMetric routerMetric = new RouterMetric();
    }

    public long getLastCheckTimestamp() {
        return lastCheckTimestamp;
    }

    public void setLastCheckTimestamp(long lastCheckTimestamp) {
        this.lastCheckTimestamp = lastCheckTimestamp;
    }

    public AtomicLong getSourceReceiveBytesCount() {
        return sourceReceiveBytesCount;
    }

    public void setSourceReceiveBytesCount(AtomicLong sourceReceiveBytesCount) {
        this.sourceReceiveBytesCount = sourceReceiveBytesCount;
    }

    public AtomicLong getSourceSendBytesCount() {
        return sourceSendBytesCount;
    }

    public void setSourceSendBytesCount(AtomicLong sourceSendBytesCount) {
        this.sourceSendBytesCount = sourceSendBytesCount;
    }

    public AtomicLong getSinkReceiveBytesCount() {
        return sinkReceiveBytesCount;
    }

    public void setSinkReceiveBytesCount(AtomicLong sinkReceiveBytesCount) {
        this.sinkReceiveBytesCount = sinkReceiveBytesCount;
    }

    public AtomicLong getSinkSendBytesCount() {
        return sinkSendBytesCount;
    }

    public void setSinkSendBytesCount(AtomicLong sinkSendBytesCount) {
        this.sinkSendBytesCount = sinkSendBytesCount;
    }

    public long getLastUpstreamBytesCount() {
        return lastUpstreamBytesCount;
    }

    public void setLastUpstreamBytesCount(long lastUpstreamBytesCount) {
        this.lastUpstreamBytesCount = lastUpstreamBytesCount;
    }

    public long getLastDownstreamBytesCount() {
        return lastDownstreamBytesCount;
    }

    public void setLastDownstreamBytesCount(long lastDownstreamBytesCount) {
        this.lastDownstreamBytesCount = lastDownstreamBytesCount;
    }

    public long addSourceReceive(int size) {
        return sourceReceiveBytesCount.addAndGet(size);
    }

    public long addSourceSend(int size) {
        return sourceSendBytesCount.addAndGet(size);
    }

    public long addSinkReceive(int size) {
        return sinkReceiveBytesCount.addAndGet(size);
    }

    public long addSinkSend(int size) {
        return sinkSendBytesCount.addAndGet(size);
    }

    public String toString() {
        return JsonUtil.object2Json(this);
    }

}
