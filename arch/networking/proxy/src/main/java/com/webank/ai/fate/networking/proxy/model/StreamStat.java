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

package com.webank.ai.fate.networking.proxy.model;


import com.webank.ai.fate.api.networking.proxy.Proxy;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.Date;

@Component
@Scope("prototype")
public class StreamStat {
    public static final String INIT = "init";
    public static final String RUNNING = "running";
    public static final String COMPLETED = "completed";
    public static final String ERROR = "error";
    public static final String PUSH = "push";
    public static final String PULL = "pull";
    public static final String UNARY_CALL = "unary_call";
    private final long startTimestamp;
    private volatile long size;
    private long lastUpdateTimestamp;
    private Date startDate;
    private String status;
    private String operation;
    private Proxy.Metadata metadata;

    public StreamStat(Proxy.Metadata metadata, String operation) {
        long now = System.currentTimeMillis();
        size = 0L;
        startTimestamp = now;
        lastUpdateTimestamp = now;
        startDate = new Date(now);
        status = INIT;

        this.metadata = metadata;
        this.operation = operation;
    }

    public void increment(long size) {
        this.size += size;
        this.lastUpdateTimestamp = System.currentTimeMillis();
        this.status = RUNNING;
    }

    public void onError() {
        this.setStatus(ERROR);
    }

    public void onComplete() {
        this.setStatus(COMPLETED);
    }

    public boolean canBeDeleted() {
        return COMPLETED.equals(status) || ERROR.equals(status);
    }

    public long getSize() {
        return size;
    }

    public long getStartTimestamp() {
        return startTimestamp;
    }

    public long getLastUpdateTimestamp() {
        return lastUpdateTimestamp;
    }

    public Date getStartDate() {
        return startDate;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
        this.lastUpdateTimestamp = System.currentTimeMillis();
    }

    public String getOperation() {
        return operation;
    }

    public Proxy.Metadata getMetadata() {
        return metadata;
    }

    @Override
    public String toString() {
        return "StreamStat{" +
                "size=" + size +
                ", startTimestamp=" + startTimestamp +
                ", lastUpdateTimestamp=" + lastUpdateTimestamp +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof StreamStat)) return false;

        StreamStat that = (StreamStat) o;

        if (size != that.size) return false;
        if (startTimestamp != that.startTimestamp) return false;
        return lastUpdateTimestamp == that.lastUpdateTimestamp;
    }

    @Override
    public int hashCode() {
        int result = (int) (size ^ (size >>> 32));
        result = 31 * result + (int) (startTimestamp ^ (startTimestamp >>> 32));
        result = 31 * result + (int) (lastUpdateTimestamp ^ (lastUpdateTimestamp >>> 32));
        return result;
    }
}
