package com.osx.broker.token;

import com.firework.cluster.rpc.Firework;


import com.osx.core.constant.Dict;
import com.osx.core.config.MetaInfo;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.flow.*;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;
import com.osx.core.token.TokenResult;
import com.osx.core.token.TokenResultStatus;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ScheduledThreadPoolExecutor;


public class DefaultTokenService extends AbstractServiceAdaptor<Firework.TokenRequest,Firework.TokenResponse> implements TokenService {

    Logger logger = LoggerFactory.getLogger(DefaultTokenService.class);

    public DefaultTokenService(){

    }

    ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1);

    MetricReport  metricReport = new FileMetricReport(Dict.SERVICE_FIREWORK_CLUSTERMANAGER);

    @Override
    protected Firework.TokenResponse doService(Context context, InboundPackage<Firework.TokenRequest> data) {
        Firework.TokenRequest tokenRequest = data.getBody();
        TokenResult tokenResult = this.requestToken(tokenRequest.getResource(),tokenRequest.getCount(),true);
//        logger.info("resource {} require {} result {}",tokenRequest.getResource(),tokenRequest.getCount(),tokenResult);
        Firework.TokenResponse.Builder   tokenResponseBuilder = Firework.TokenResponse.newBuilder();
        tokenResponseBuilder.setStatus(tokenResult.getStatus());
        tokenResponseBuilder.setWaitInMs(tokenResult.getWaitInMs());
        return tokenResponseBuilder.build();
    }

    @Override
    protected Firework.TokenResponse transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        return null;
    }


//    Config config;
//
//    public DefaultTokenService(Config config){
//        config = config;
//    }

    @Override
    public TokenResult requestToken(String resource, int acquireCount, boolean prioritized) {

        if(StringUtils.isEmpty(resource)){
            return badRequest();
        }
        FlowRule rule = ClusterFlowRuleManager.getFlowRuleByResource(resource);
        if (rule == null) {
            logger.error("resource {} no rule",resource);
            ClusterMetric clusterMetric = ClusterMetricStatistics.getMetric(resource);
            if(clusterMetric==null){
                ClusterMetricStatistics.putMetricIfAbsent(resource,new ClusterMetric(MetaInfo.PROPERTY_SAMPLE_COUNT,MetaInfo.PROPERTY_INTERVAL_MS));
                clusterMetric =ClusterMetricStatistics.getMetric(resource);
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
      //  ConcurrentClusterFlowChecker.releaseConcurrentToken(tokenId);
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

//    @Override
//    public void onApplicationEvent(ApplicationReadyEvent event) {
//
//
//
//        executor.scheduleAtFixedRate(() -> {
//            long current = TimeUtil.currentTimeMillis();
//            List<MetricNode> reportList = Lists.newArrayList();
//            List<MetricNode> modelReportList = Lists.newArrayList();
//            ClusterMetricStatistics.getMetricMap().forEach((sourceName, clusterMetric) -> {
//                long  pass = clusterMetric.getCurrentCountForReport(ClusterFlowEvent.PASS);
////                FlowCounter successCounter = successMap.get(sourceName);
////                FlowCounter blockCounter = blockMap.get(sourceName);
////                FlowCounter exceptionCounter = exceptionMap.get(sourceName);
//                MetricNode metricNode = new MetricNode();
//                metricNode.setTimestamp(current);
//                metricNode.setResource(sourceName);
//                metricNode.setPassQps(pass);
//                metricNode.setBlockQps(0);
//                metricNode.setExceptionQps( 0);
//                metricNode.setSuccessQps(pass);
//                reportList.add(metricNode);
//
//            });
////            logger.info("try to report {}",reportList);
//            metricReport.report(reportList);
//        }, 0, 1, TimeUnit.SECONDS);
//    }
}
