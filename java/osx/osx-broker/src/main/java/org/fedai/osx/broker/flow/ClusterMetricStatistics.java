package org.fedai.osx.broker.flow;


import org.fedai.osx.core.flow.ClusterMetric;
import org.fedai.osx.core.utils.AssertUtil;

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
