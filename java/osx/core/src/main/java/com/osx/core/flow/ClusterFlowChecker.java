package com.osx.core.flow;


import com.osx.core.token.TokenResult;
import com.osx.core.token.TokenResultStatus;


final public class ClusterFlowChecker {

    private ClusterFlowChecker() {
    }

    private static double calcGlobalThreshold(FlowRule rule) {
        double count = rule.getCount();
//        switch (rule.getClusterConfig().getThresholdType()) {
//            case ClusterRuleConstant.FLOW_THRESHOLD_GLOBAL:
//                return count;
//            case ClusterRuleConstant.FLOW_THRESHOLD_AVG_LOCAL:
//            default:
//                int connectedCount = ClusterFlowRuleManager.getConnectedCount(rule.getClusterConfig().getFlowId());
//                return count * connectedCount;
//        }

        return count;
    }

    static boolean allowProceed(long flowId) {
        String namespace = ClusterFlowRuleManager.getNamespace(flowId);
        return GlobalRequestLimiter.tryPass(namespace);
    }

    static public TokenResult acquireClusterToken(/*@Valid*/ FlowRule rule, int acquireCount, boolean prioritized) {
//        Long id = rule.getClusterConfig().getFlowId();
//
//        if (!allowProceed(id)) {
//            return new TokenResult(TokenResultStatus.TOO_MANY_REQUEST);
//        }

        ClusterMetric metric = ClusterMetricStatistics.getMetric(rule.getResource());
        if (metric == null) {
            return new TokenResult(TokenResultStatus.FAIL);
        }

        double latestQps = metric.getAvg(ClusterFlowEvent.PASS);
        double globalThreshold = calcGlobalThreshold(rule);
        double nextRemaining = globalThreshold - latestQps - acquireCount;

        if (nextRemaining >= 0) {
            // TODO: checking logic and metric operation should be separated.
            metric.add(ClusterFlowEvent.PASS, acquireCount);
            metric.add(ClusterFlowEvent.PASS_REQUEST, 1);
            if (prioritized) {
                // Add prioritized pass.
                metric.add(ClusterFlowEvent.OCCUPIED_PASS, acquireCount);
            }
            // Remaining count is cut down to a smaller integer.
            return new TokenResult(TokenResultStatus.OK)
                    .setRemaining((int) nextRemaining)
                    .setWaitInMs(0);
        } else {
            if (prioritized) {
                // Try to occupy incoming buckets.
                double occupyAvg = metric.getAvg(ClusterFlowEvent.WAITING);
                if (occupyAvg <= 0.8 * globalThreshold) {
                    int waitInMs = metric.tryOccupyNext(ClusterFlowEvent.PASS, acquireCount, globalThreshold);
                    // waitInMs > 0 indicates pre-occupy incoming buckets successfully.
                    if (waitInMs > 0) {
                        //ClusterServerStatLogUtil.log("flow|waiting|" + id);
                        return new TokenResult(TokenResultStatus.SHOULD_WAIT)
                                .setRemaining(0)
                                .setWaitInMs(waitInMs);
                    }
                    // Or else occupy failed, should be blocked.
                }
            }
            // Blocked.
            metric.add(ClusterFlowEvent.BLOCK, acquireCount);
            metric.add(ClusterFlowEvent.BLOCK_REQUEST, 1);
            //    ClusterServerStatLogUtil.log("flow|block|" + id, acquireCount);
            //    ClusterServerStatLogUtil.log("flow|block_request|" + id, 1);
            if (prioritized) {
                // Add prioritized block.
                metric.add(ClusterFlowEvent.OCCUPIED_BLOCK, acquireCount);
                //       ClusterServerStatLogUtil.log("flow|occupied_block|" + id, 1);
            }

            return blockedResult(rule);
        }
    }

    private static TokenResult blockedResult(FlowRule rule) {
        return new TokenResult(TokenResultStatus.BLOCKED)
                .setRemaining(0)
                .setWaitInMs(0)
                .setWaitInMs(rule.getMaxQueueingTimeMs());
    }
}
