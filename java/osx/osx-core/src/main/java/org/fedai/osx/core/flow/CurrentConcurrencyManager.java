package org.fedai.osx.core.flow;


import org.fedai.osx.core.datasource.NamedThreadFactory;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicInteger;


public final class CurrentConcurrencyManager {
    /**
     * use ConcurrentHashMap to store the nowCalls of rules.
     */
    private static final ConcurrentHashMap<Long, AtomicInteger> NOW_CALLS_MAP = new ConcurrentHashMap<Long, AtomicInteger>();

    @SuppressWarnings("PMD.ThreadPoolCreationRule")
    private static final ScheduledExecutorService SCHEDULER = Executors.newScheduledThreadPool(1,
            new NamedThreadFactory("sentinel-cluster-concurrency-record-task", true));

//    static {
//        ClusterConcurrentCheckerLogListener logTask = new ClusterConcurrentCheckerLogListener();
//        SCHEDULER.scheduleAtFixedRate(logTask, 0, 1, TimeUnit.SECONDS);
//    }

    /**
     * ,
     * add current concurrency.
     */
    public static void addConcurrency(Long flowId, Integer acquireCount) {

        AtomicInteger nowCalls = NOW_CALLS_MAP.get(flowId);
        if (nowCalls == null) {
            return;
        }
        nowCalls.getAndAdd(acquireCount);
    }

    /**
     * get the current concurrency.
     */
    public static AtomicInteger get(Long flowId) {
        return NOW_CALLS_MAP.get(flowId);
    }

    /**
     * delete the current concurrency.
     */
    public static void remove(Long flowId) {
        NOW_CALLS_MAP.remove(flowId);
    }

    /**
     * put the current concurrency.
     */
    public static void put(Long flowId, Integer nowCalls) {
        NOW_CALLS_MAP.put(flowId, new AtomicInteger(nowCalls));
    }

    /**
     * check flow id.
     */
    public static boolean containsFlowId(Long flowId) {
        return NOW_CALLS_MAP.containsKey(flowId);
    }

    /**
     * get NOW_CALLS_MAP.
     */
    public static Set<Long> getConcurrencyMapKeySet() {
        return NOW_CALLS_MAP.keySet();
    }
}
