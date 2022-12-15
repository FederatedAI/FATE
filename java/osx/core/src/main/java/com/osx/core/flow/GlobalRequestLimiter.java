
package com.osx.core.flow;

import com.osx.core.utils.AssertUtil;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;



public final class GlobalRequestLimiter {

    private static final Map<String, RequestLimiter> GLOBAL_QPS_LIMITER_MAP = new ConcurrentHashMap<>();

    public static void initIfAbsent(String namespace) {
        AssertUtil.notEmpty(namespace, "namespace cannot be empty");
        if (!GLOBAL_QPS_LIMITER_MAP.containsKey(namespace)) {
            GLOBAL_QPS_LIMITER_MAP.put(namespace, new RequestLimiter(10000));
        }
    }

    public static RequestLimiter getRequestLimiter(String namespace) {
        if (namespace == null) {
            return null;
        }
        return GLOBAL_QPS_LIMITER_MAP.get(namespace);
    }

    public static boolean tryPass(String namespace) {
        if (namespace == null) {
            return false;
        }
        RequestLimiter limiter = GLOBAL_QPS_LIMITER_MAP.get(namespace);
        if (limiter == null) {
            return true;
        }
        return limiter.tryPass();
    }

    public static double getCurrentQps(String namespace) {
        RequestLimiter limiter = getRequestLimiter(namespace);
        if (limiter == null) {
            return 0;
        }
        return limiter.getQps();
    }

    public static double getMaxAllowedQps(String namespace) {
        RequestLimiter limiter = getRequestLimiter(namespace);
        if (limiter == null) {
            return 0;
        }
        return limiter.getQpsAllowed();
    }

    public static void applyMaxQpsChange(double maxAllowedQps) {
        AssertUtil.isTrue(maxAllowedQps >= 0, "max allowed QPS should > 0");
        for (RequestLimiter limiter : GLOBAL_QPS_LIMITER_MAP.values()) {
            if (limiter != null) {
                limiter.setQpsAllowed(maxAllowedQps);
            }
        }
    }

    private GlobalRequestLimiter() {}
}
