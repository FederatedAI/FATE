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

package com.webank.ai.fate.networking.proxy.manager;


import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.model.StreamStat;
import com.webank.ai.fate.networking.proxy.util.ToStringUtils;
import io.grpc.netty.shaded.io.netty.util.internal.ConcurrentSet;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.commons.text.StringSubstitutor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

@Component
public class StatsManager {
    private static final String LOG_TEMPLATE;
    private static final Logger LOGGER = LogManager.getLogger("stat");

    static {
        LOG_TEMPLATE = "streaming stat: Proxy.Metadata [${metadata}], " +
                "bytes sent: ${bytesSent}, " +
                "avg speed: ${avgSpeed} byte/s, " +
                "last updated: ${lastUpdateTime}, " +
                "started: ${startTime}, " +
                "time eclapsed: ${secEclapsed}, " +
                "status: ${status}, " +
                "finished: ${isFinished}";
    }

    @Autowired
    private ToStringUtils toStringUtils;
    private ConcurrentSet<StreamStat> streamStats;
    private ReadWriteLock streamStatsRwLock;
    private NumberFormat numberFormat;
    private SimpleDateFormat simpleDateFormat;

    public StatsManager() {
        this.streamStats = new ConcurrentSet<>();
        this.streamStatsRwLock = new ReentrantReadWriteLock();

        this.numberFormat = NumberFormat.getNumberInstance(Locale.CHINA);
        this.simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss,SSS");
    }

    public void add(StreamStat streamStat) {
        try {
            streamStatsRwLock.writeLock().lock();
            streamStats.add(streamStat);
        } catch (Exception e) {
            LOGGER.error("unexpected error: {}", ExceptionUtils.getStackTrace(e));
        } finally {
            streamStatsRwLock.writeLock().unlock();
        }
    }

    public void logAllStatus() {
        LOGGER.info("------------ streaming stat ------------");

        if (streamStats.size() <= 0) {
            return;
        }
        long now = System.currentTimeMillis();

        ConcurrentSet<StreamStat> oldStreamStats = null;
        try {
            streamStatsRwLock.writeLock().lock();
            oldStreamStats = streamStats;
            streamStats = new ConcurrentSet<>();
        } catch (Exception e) {
            LOGGER.error(ExceptionUtils.getStackTrace(e));
        } finally {
            streamStatsRwLock.writeLock().unlock();
        }

        Map<String, String> valuesMap = new ConcurrentHashMap<>(10);
        StringSubstitutor stringSubstitutor = new StringSubstitutor(valuesMap);

        int logCount = 0;
        int notFinishedCount = 0;
        int notRemovedCount = 0;

        for (StreamStat streamStat : oldStreamStats) {
            valuesMap.clear();

            Proxy.Metadata metadata = streamStat.getMetadata();

            valuesMap.put("metadata", toStringUtils.toOneLineString(metadata));
            valuesMap.put("bytesSent", numberFormat.format(streamStat.getSize()));
            valuesMap.put("lastUpdateTime", simpleDateFormat.format(new Date(streamStat.getLastUpdateTimestamp())));
            valuesMap.put("startTime", simpleDateFormat.format(streamStat.getStartDate()));

            long avgSpeedTimeline = 0;

            if (streamStat.canBeDeleted()) {
                valuesMap.put("isFinished", "true");
                avgSpeedTimeline = streamStat.getLastUpdateTimestamp();
            } else {
                valuesMap.put("isFinished", "false");
                avgSpeedTimeline = now;
                ++notFinishedCount;

                if (now - streamStat.getLastUpdateTimestamp() <= 60000) {
                    streamStats.add(streamStat);
                    ++notRemovedCount;
                }
            }

            long secEclapsed = (avgSpeedTimeline - streamStat.getStartTimestamp()) / 1000;
            if (secEclapsed <= 0) {
                secEclapsed = 1;
            }

            valuesMap.put("secEclapsed", numberFormat.format(secEclapsed));
            valuesMap.put("avgSpeed", numberFormat.format(streamStat.getSize() / secEclapsed));
            valuesMap.put("status", streamStat.getStatus());

            String log = stringSubstitutor.replace(LOG_TEMPLATE);
            LOGGER.info(log);
            ++logCount;
        }

        LOGGER.info("stream logCount: {}, not finished: {}, not removed: {}",
                logCount, notFinishedCount, notRemovedCount);
    }
}
