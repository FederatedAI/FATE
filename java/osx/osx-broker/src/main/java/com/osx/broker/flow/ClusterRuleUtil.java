package com.osx.broker.flow;

public final class ClusterRuleUtil {

    private ClusterRuleUtil() {
    }

    public static boolean validId(Long id) {
        return id != null && id > 0;
    }
}
