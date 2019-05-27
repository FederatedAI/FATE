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

package com.webank.ai.fate.networking.proxy.util;

import com.google.protobuf.Descriptors;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Component;

@Component
public class Timeouts {
    public static final long MAX_OVERALL_TIMEOUT = 12L * 3600 * 1000;                // 12h
    public static final long DEFAULT_OVERALL_TIMEOUT = MAX_OVERALL_TIMEOUT;

    public static final long MAX_COMPLETION_WAIT_TIMEOUT = MAX_OVERALL_TIMEOUT;
    public static final long DEFAULT_COMPLETION_WAIT_TIMEOUT = 60L * 60 * 1000;      // 60m

    public static final long MAX_PACKET_INTERVAL_TIMEOUT = MAX_OVERALL_TIMEOUT;
    public static final long DEFAULT_PACKET_INTERVAL_TIMEOUT = 20L * 1000;           // 20s

    private static final Logger LOGGER = LogManager.getLogger(Timeouts.class);

    public boolean isTimeout(long timeout, long startTimestamp) {
        return isTimeout(timeout, startTimestamp, System.currentTimeMillis());
    }

    public boolean isTimeout(long timeout, long startTimestamp, long endTimestamp) {
        /*LOGGER.info("timeout: {}, start: {}, end: {}, cal: {}, result: {}",
                timeout, startTimestamp, endTimestamp, (endTimestamp - startTimestamp),
                ((endTimestamp - startTimestamp) > timeout));*/
        return (endTimestamp - startTimestamp) > timeout;
    }

    public long getOverallTimeout(Proxy.Metadata metadata) {
        long result = DEFAULT_OVERALL_TIMEOUT;
        if (metadata != null && metadata.hasConf()) {
            Proxy.Conf conf = metadata.getConf();
            long proposedValue = conf.getOverallTimeout();

            result = getValue(proposedValue, DEFAULT_OVERALL_TIMEOUT, 1, MAX_OVERALL_TIMEOUT);
        }

        return result;
    }

    public long getCompletionWaitTimeout(Proxy.Metadata metadata) {
        long result = DEFAULT_COMPLETION_WAIT_TIMEOUT;
        if (metadata != null && metadata.hasConf()) {
            Proxy.Conf conf = metadata.getConf();
            long proposedValue = conf.getCompletionWaitTimeout();

            result = getValue(proposedValue, DEFAULT_COMPLETION_WAIT_TIMEOUT, 1, MAX_COMPLETION_WAIT_TIMEOUT);
        }

        return result;
    }

    public long getPacketIntervalTimeout(Proxy.Metadata metadata) {
        long result = DEFAULT_PACKET_INTERVAL_TIMEOUT;
        if (metadata != null && metadata.hasConf()) {
            Proxy.Conf conf = metadata.getConf();
            long proposedValue = conf.getPacketIntervalTimeout();

            result = getValue(proposedValue, DEFAULT_PACKET_INTERVAL_TIMEOUT, 1, MAX_PACKET_INTERVAL_TIMEOUT);
        }

        return result;
    }

    private long getTimeout(Proxy.Metadata metadata, String timeoutFieldName, long defaultValue, long minValue, long maxValue) {
        long result = defaultValue;
        if (metadata != null && metadata.hasConf()) {
            Proxy.Conf conf = metadata.getConf();

            Descriptors.Descriptor descriptor = Proxy.Conf.getDescriptor();
            Descriptors.FieldDescriptor fieldDescriptor = descriptor.findFieldByName(timeoutFieldName);

            long proposedValue = (long) conf.getField(fieldDescriptor);

            result = getValue(proposedValue, defaultValue, minValue, maxValue);
        }

        return result;
    }

    private long getValue(long proposedValue, long defaultValue, long minValue, long maxValue) {
        long result = proposedValue;
        // LOGGER.info("proposed: {}, default: {}, min: {}, max: {}", proposedValue, defaultValue, minValue, maxValue);

        if (proposedValue < minValue) {
            result = defaultValue;
        } else if (proposedValue > maxValue) {
            result = maxValue;
        }

        return result;
    }
}
