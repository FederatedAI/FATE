package com.osx.core.flow;

import java.util.concurrent.atomic.LongAdder;


public class ClusterMetricBucket {

    private final LongAdder[] counters;

    public ClusterMetricBucket() {
        ClusterFlowEvent[] events = ClusterFlowEvent.values();
        this.counters = new LongAdder[events.length];
        for (ClusterFlowEvent event : events) {
            counters[event.ordinal()] = new LongAdder();
        }
    }

    public void reset() {
        for (ClusterFlowEvent event : ClusterFlowEvent.values()) {
            counters[event.ordinal()].reset();
        }
    }

    public long get(ClusterFlowEvent event) {
        return counters[event.ordinal()].sum();
    }

    public ClusterMetricBucket add(ClusterFlowEvent event, long count) {
        counters[event.ordinal()].add(count);
        return this;
    }
}
