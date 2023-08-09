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
package org.fedai.osx.broker.token;

import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.flow.*;
import org.fedai.osx.core.token.TokenResult;
import org.fedai.osx.core.token.TokenResultStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ScheduledThreadPoolExecutor;


public class DefaultTokenService implements TokenService {

    Logger logger = LoggerFactory.getLogger(DefaultTokenService.class);
    ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1);
    MetricReport metricReport = new FileMetricReport(Dict.SERVICE_OSX_CLUSTERMANAGER);
    public DefaultTokenService() {

    }
    @Override
    public TokenResult requestToken(String resource, int acquireCount, boolean prioritized) {

        if (StringUtils.isEmpty(resource)) {
            return badRequest();
        }
        FlowRule rule = ClusterFlowRuleManager.getFlowRuleByResource(resource);
        if (rule == null) {
            //logger.error("resource {} no rule", resource);
            ClusterMetric clusterMetric = ClusterMetricStatistics.getMetric(resource);
            if (clusterMetric == null) {
                ClusterMetricStatistics.putMetricIfAbsent(resource, new ClusterMetric(MetaInfo.PROPERTY_FLOW_CONTROL_SAMPLE_COUNT, MetaInfo.PROPERTY_FLOW_CONTROL_SAMPLE_INTERVAL));
                clusterMetric = ClusterMetricStatistics.getMetric(resource);
            }
            clusterMetric.add(ClusterFlowEvent.PASS, acquireCount);
            clusterMetric.add(ClusterFlowEvent.PASS_REQUEST, 1);
            if (prioritized) {
                clusterMetric.add(ClusterFlowEvent.OCCUPIED_PASS, acquireCount);
            }
            return new TokenResult(TokenResultStatus.NO_RULE_EXISTS);
        }
        return ClusterFlowChecker.acquireClusterToken(rule, acquireCount, prioritized);
    }
    @Override
    public void releaseConcurrentToken(Long tokenId) {
        if (tokenId == null) {
            return;
        }
    }
    private boolean notValidRequest(Long id, int count) {
        return id == null || id <= 0 || count <= 0;
    }

    private boolean notValidRequest(String address, Long id, int count) {
        return address == null || "".equals(address) || id == null || id <= 0 || count <= 0;
    }
    private TokenResult badRequest() {
        return new TokenResult(TokenResultStatus.BAD_REQUEST);
    }

}
