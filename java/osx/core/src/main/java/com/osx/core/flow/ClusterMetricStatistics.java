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
package com.osx.core.flow;

import com.osx.core.utils.AssertUtil;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class ClusterMetricStatistics {

    private static final Map<String, ClusterMetric> METRIC_MAP = new ConcurrentHashMap<>();

    private ClusterMetricStatistics() {
    }

    public static void clear() {
        METRIC_MAP.clear();
    }

    public static void putMetric(String resource, ClusterMetric metric) {
        AssertUtil.notNull(metric, "Cluster metric cannot be null");
        METRIC_MAP.put(resource, metric);
    }

    public static boolean putMetricIfAbsent(String resource, ClusterMetric metric) {
        AssertUtil.notNull(metric, "Cluster metric cannot be null");
        if (METRIC_MAP.containsKey(resource)) {
            return false;
        }
        METRIC_MAP.put(resource, metric);
        return true;
    }

    public static void removeMetric(String resource) {
        METRIC_MAP.remove(resource);
    }

    public static ClusterMetric getMetric(String resource) {
        return METRIC_MAP.get(resource);
    }

//    public static void resetFlowMetrics() {
//        Set<Long> keySet = METRIC_MAP.keySet();
//        for (Long id : keySet) {
//            METRIC_MAP.put(id, new ClusterMetric(ClusterServerConfigManager.getSampleCount(),
//                ClusterServerConfigManager.getIntervalMs()));
//        }
//    }

    public static Map<String, ClusterMetric> getMetricMap() {
        return METRIC_MAP;
    }


}
