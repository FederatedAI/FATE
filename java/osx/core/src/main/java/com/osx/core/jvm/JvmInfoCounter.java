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

package com.osx.core.jvm;

import com.google.common.collect.Lists;
import com.osx.core.flow.LeapArray;
import com.osx.core.flow.TimeUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class JvmInfoCounter {

    static boolean started = false;
    static ScheduledThreadPoolExecutor executorService = new ScheduledThreadPoolExecutor(1);
    private static LeapArray<JvmInfo> data = new JvmInfoLeapArray(10, 10000);
    private static Logger logger = LoggerFactory.getLogger(JvmInfoCounter.class);

    public static List<JvmInfo> getMemInfos() {
        List<JvmInfo> result = Lists.newArrayList();
        if (data.listAll() != null) {
            data.listAll().forEach(window -> {
                result.add(window.value());
            });
        }
        return result;
    }

    public static synchronized void start() {
        if (!started) {
            executorService.scheduleAtFixedRate(new Runnable() {
                @Override
                public void run() {
                    long timestamp = TimeUtil.currentTimeMillis();
                    JvmInfo memInfo = data.currentWindow().value();
                    memInfo.heap = JVMMemoryUtils.getHeapMemoryUsage();
                    memInfo.old = JVMMemoryUtils.getOldGenMemoryUsage();
                    memInfo.eden = JVMMemoryUtils.getEdenSpaceMemoryUsage();
                    memInfo.nonHeap = JVMMemoryUtils.getNonHeapMemoryUsage();
                    memInfo.survivor = JVMMemoryUtils.getSurvivorSpaceMemoryUsage();
                    memInfo.yongGcCount = JVMGCUtils.getYoungGCCollectionCount();
                    memInfo.yongGcTime = JVMGCUtils.getYoungGCCollectionTime();
                    memInfo.fullGcCount = JVMGCUtils.getFullGCCollectionCount();
                    memInfo.fullGcTime = JVMGCUtils.getFullGCCollectionTime();
                    memInfo.threadCount = JVMThreadUtils.getThreadCount();
                    memInfo.timestamp = timestamp;
                }
            }, 0, 1000, TimeUnit.MILLISECONDS);
            started = true;
        }
    }



}
